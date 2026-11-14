"""Proof of concept: train a VLA backbone too big for one card across lab's two GPUs.

Three ways to run the same model, picked with STRATEGY:
  single  everything on one card (the 3090), to show where one card runs out
  split   the decoder layers divided between the cards in proportion to their memory;
          one card works while the other waits, but the cards needn't match
  fsdp    one process per card, each holding a shard of every weight and running its own
          batch (data parallel); weights are gathered layer by layer for compute. Every card
          holds the same share, so the smallest card sets the limit, and the slowest one the pace

Other knobs (env vars): MODEL=2.2B|500M, STEPS, BATCH (per card), SHARDS (dataset shards to load),
DATA (shard directory), CHECKPOINT=1 for activation checkpointing, OFFLOAD=1 to keep fsdp's weights and Adam state in CPU RAM.
Prints step time, throughput and peak memory per card, or which card ran out.
"""
import glob
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # torch's cuda:N is nvidia-smi's GPU N
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForImageTextToText, AutoProcessor

import frames
from dataset import Dataset
from tasks import sample_phrases
from vlm import JOINT_MAX, JOINT_MIN, RobotControlOutputHead

MODELS = {
    "500M": ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "7b375e1b73b11138ff12fe22c8f2822d8fe03467", 960),
    "2.2B": ("HuggingFaceTB/SmolVLM2-2.2B-Instruct", "be92141c8d0ceb98596a923c308ac2bbb4a7c466", 2048),
}


def env(name, default):
    return type(default)(os.environ.get(name, default))


def move_inputs_hook(module, args, kwargs):
    """Forward pre-hook: bring every tensor argument to the device this module's weights live on."""
    device = next(module.parameters()).device

    def move(x):
        if torch.is_tensor(x):
            return x.to(device)
        if isinstance(x, tuple):
            return tuple(move(v) for v in x)
        if isinstance(x, dict):
            return {k: move(v) for k, v in x.items()}
        return x
    return tuple(move(a) for a in args), {k: move(v) for k, v in kwargs.items()}


class PocTrainer:

    def __init__(self, strategy="single", model_name="2.2B", steps=30, warmup=5, batch_size=4,
                 shards=20, chunk_size=8, n_actions=9, learning_rate=1e-5, checkpointing=False,
                 offload=False, big_gpu=1, small_gpu=0, data_path="dataset"):
        self.strategy = strategy
        self.model_name = model_name
        self.steps = steps
        self.warmup = warmup  # steps left out of the timing: allocator and kernel warmup
        self.batch_size = batch_size
        self.shards = shards
        self.chunk_size = chunk_size
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.checkpointing = checkpointing
        self.offload = offload
        self.big_gpu = big_gpu  # the 3090 under PCI_BUS_ID order
        self.small_gpu = small_gpu  # the 3060
        self.data_path = data_path

    def load_data(self):
        """A few shards through the normal loader: enough for real images and actions, small enough to load fast."""
        files = sorted(glob.glob(os.path.join(self.data_path, "**", "*.npz"), recursive=True))[:self.shards]
        assert files, f"no shards under {self.data_path}"
        subset = tempfile.mkdtemp(prefix="poc_shards_")
        for f in files:
            os.symlink(os.path.abspath(f), os.path.join(subset, os.path.basename(f)))
        dataset = Dataset(max_size=400 * len(files), image_size=frames.VLM_SIZE, n_actions=self.n_actions,
                          n_joints=len(JOINT_MIN))
        dataset.load_data(subset)
        return dataset

    def build_model(self):
        """The VLA: pretrained backbone with the decoder layers trainable in fp32, vision and embeddings frozen in bf16."""
        model_id, revision, emb_dim = MODELS[self.model_name]
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.processor.tokenizer.padding_side = "left"
        model = AutoModelForImageTextToText.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16)
        for p in model.parameters():
            p.requires_grad = False
        text = model.model.text_model
        text.float()
        for p in text.parameters():
            p.requires_grad = True
        # Embeddings stay frozen bf16: image features are spliced into them in place, so the dtypes must match.
        text.embed_tokens.to(torch.bfloat16).requires_grad_(False)
        model.lm_head = RobotControlOutputHead(vla_output_dim=emb_dim, n_actions=self.n_actions * self.chunk_size)
        if self.checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"{self.model_name}: {total / 1e9:.2f}B params, {trainable / 1e9:.2f}B trainable "
              f"(~{trainable * 16 / 1e9:.0f} GB for fp32 weights, grads and Adam state)", flush=True)
        return model

    def place_split(self, model):
        """Vision, embeddings and the first layers on the small card, the rest on the big one, by memory share."""
        small, big = torch.device(f"cuda:{self.small_gpu}"), torch.device(f"cuda:{self.big_gpu}")
        layers = model.model.text_model.layers
        mem_small = torch.cuda.get_device_properties(small).total_memory
        mem_big = torch.cuda.get_device_properties(big).total_memory
        # One layer fewer on the small card: it also holds the embedding table and vision tower.
        cut = max(1, round(len(layers) * mem_small / (mem_small + mem_big)) - 1)
        model.to(small)
        for module in [*layers[cut:], model.model.text_model.norm, model.lm_head]:
            module.to(big)
            module.register_forward_pre_hook(move_inputs_hook, with_kwargs=True)
        print(f"split: layers 0-{cut - 1} + vision + embeddings on cuda:{self.small_gpu}, "
              f"layers {cut}-{len(layers) - 1} + head on cuda:{self.big_gpu}", flush=True)
        return small  # where inputs go in

    def place_fsdp(self, model, rank):
        from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
        device = torch.device(f"cuda:{rank}")
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        offload_policy = CPUOffloadPolicy() if self.offload else None
        if not self.offload:
            model.to(device)
        for module in [*model.model.text_model.layers, model.model.vision_model, model]:
            if offload_policy:
                fully_shard(module, mp_policy=mp_policy, offload_policy=offload_policy)
            else:
                fully_shard(module, mp_policy=mp_policy)
        return device

    def prompts_with_joints(self, prompts, joint_pos):
        bins = ((joint_pos - JOINT_MIN) / (JOINT_MAX - JOINT_MIN) * 99).round().clamp(0, 99).int().tolist()
        return [f"{p}. Joints: {' '.join(map(str, b))}" for p, b in zip(prompts, bins)]

    def batch(self, dataset, device):
        states, actions, mask, tasks = dataset.sample_batch(self.batch_size, self.chunk_size)
        prompts = self.prompts_with_joints(sample_phrases(tasks), torch.tensor(states["joint_pos"]))
        messages = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}] for p in prompts]
        text = [self.processor.apply_chat_template(m, add_generation_prompt=True,
                                                   processor_kwargs={"do_image_splitting": False}) for m in messages]
        inputs = self.processor(text=text, images=[[img] for img in states["camera_scene"]], return_tensors="pt",
                                padding=True, do_image_splitting=False)
        inputs = inputs.to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        return inputs, torch.tensor(actions, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

    def loss(self, model, inputs, actions, mask):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**inputs).logits[:, -1, :]
        pred = logits.float().view(len(actions), self.chunk_size, self.n_actions)
        actions, mask = actions.to(pred.device), mask.to(pred.device)
        return (((pred - actions) ** 2).mean(-1) * mask).sum() / mask.sum()

    def devices(self):
        return sorted({self.small_gpu, self.big_gpu}) if self.strategy != "single" else [self.big_gpu]

    def train(self, rank=0, world=1):
        torch.manual_seed(123)
        np.random.seed(123 + rank)  # fsdp ranks draw different batches
        dataset = self.load_data()
        model = self.build_model()
        if self.strategy == "single":
            device = torch.device(f"cuda:{self.big_gpu}")
            model.to(device)
        elif self.strategy == "split":
            device = self.place_split(model)
        else:
            device = self.place_fsdp(model, rank)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=self.learning_rate)
        for d in self.devices():
            torch.cuda.reset_peak_memory_stats(d)

        times, losses = [], []
        for step in range(self.steps):
            start = time.time()
            inputs, actions, mask = self.batch(dataset, device)
            loss = self.loss(model, inputs, actions, mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            for d in self.devices():
                torch.cuda.synchronize(d)
            if step >= self.warmup:
                times.append(time.time() - start)
            losses.append(loss.item())
            if rank == 0 and step % 5 == 0:
                print(f"step {step} loss {loss.item():.4f} {time.time() - start:.2f}s", flush=True)
        return times, losses

    def peak_memory(self, devices):
        return {d: (torch.cuda.get_device_name(d), torch.cuda.max_memory_allocated(d) / 2**30,
                    torch.cuda.get_device_properties(d).total_memory / 2**30) for d in devices}

    def report(self, times, losses, peaks, world):
        step = float(np.mean(times))
        print("=" * 60)
        print(f"RESULT strategy={self.strategy} model={self.model_name} batch/card={self.batch_size} "
              f"checkpointing={self.checkpointing} offload={self.offload}")
        print(f"  step {step:.2f}s, {self.batch_size * world / step:.1f} samples/s, "
              f"loss {losses[0]:.4f} -> {np.mean(losses[-5:]):.4f}")
        for d, (name, peak, total) in sorted(peaks.items()):
            print(f"  cuda:{d} {name}: peak {peak:.1f} of {total:.1f} GiB")
        print("=" * 60, flush=True)


def check_cards(trainer):
    """Fail loudly if cuda:N isn't the card we think it is (the 3060/3090 swap trap)."""
    names = {i: torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())}
    print(f"cards: {names}", flush=True)
    if len(names) > 1:
        assert "3090" in names[trainer.big_gpu] and "3060" in names[trainer.small_gpu], names


def fsdp_worker(rank, world, trainer):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29511")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device(f"cuda:{rank}"))
    try:
        times, losses = trainer.train(rank, world)
        peaks = [None] * world
        dist.all_gather_object(peaks, trainer.peak_memory([rank]))
        if rank == 0:
            trainer.report(times, losses, {k: v for p in peaks for k, v in p.items()}, world)
    except torch.OutOfMemoryError:
        print(f"OOM on cuda:{rank} {torch.cuda.get_device_name(rank)}", flush=True)
        raise
    finally:
        dist.destroy_process_group()


def main():
    trainer = PocTrainer(strategy=env("STRATEGY", "single"), model_name=env("MODEL", "2.2B"),
                         steps=env("STEPS", 30), batch_size=env("BATCH", 4), shards=env("SHARDS", 20),
                         checkpointing=env("CHECKPOINT", 0) == 1, offload=env("OFFLOAD", 0) == 1,
                         big_gpu=env("BIG_GPU", 1), small_gpu=env("SMALL_GPU", 0), data_path=env("DATA", "dataset"))
    check_cards(trainer)
    if trainer.strategy == "fsdp":
        world = torch.cuda.device_count()
        mp.spawn(fsdp_worker, args=(world, trainer), nprocs=world)
        return
    try:
        times, losses = trainer.train()
    except torch.OutOfMemoryError as e:
        print(f"OOM: {str(e).splitlines()[0]}", flush=True)
        for d in trainer.devices():
            print(f"  cuda:{d} {torch.cuda.get_device_name(d)}: {torch.cuda.max_memory_allocated(d) / 2**30:.1f} GiB "
                  f"allocated at peak", flush=True)
        sys.exit(1)
    trainer.report(times, losses, trainer.peak_memory(trainer.devices()), 1)


if __name__ == "__main__":
    main()
