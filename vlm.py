import os
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch.nn as nn
from torchinfo import summary


# Per-joint min/max over the dataset; fingers (last two) span only 0-0.049 m, so each joint is binned to its own range.
JOINT_MIN = torch.tensor([-2.9014, -1.7744, -2.3526, -3.0786, -0.7136, -0.0261, -0.0030, -0.0026, -0.0027])
JOINT_MAX = torch.tensor([ 2.2670, -0.9490,  2.5636, -0.3810,  1.5339,  2.1493,  2.5794,  0.0489,  0.0428])


class VisionLanguageModel:

    def __init__(self, vlm_config, num_actions, chunk_size=1, done_hidden_dim=256):

        torch.manual_seed(123)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = vlm_config
        self.num_actions = num_actions
        self.chunk_size = chunk_size
        self.processor = AutoProcessor.from_pretrained(
                self.config["model_id"],
                revision=self.config["revision"]
                )

        self.processor.tokenizer.padding_side = "left"

        self.model = AutoModelForImageTextToText.from_pretrained(
                self.config["model_id"],
                revision=self.config["revision"],
                dtype=getattr(torch, self.config["dtype"])
                )
       

        for param in self.model.parameters():
            param.requires_grad = False

        # Trained weights need fp32 so Adam's small updates aren't rounded away; autocast below bridges the dtypes.
        self.model.model.text_model.layers[-4:].float()
        for param in self.model.model.text_model.layers[-4:].parameters():
            param.requires_grad = True

        # One flat output per action in the chunk; get_action_logits reshapes it to (chunk, actions).
        self.model.lm_head = RobotControlOutputHead(vla_output_dim=self.config["emb_dim"],
                                                    n_actions=num_actions * chunk_size)
        
        self.model.to(self.device)

        # Sits beside self.model, not in it, so action training and checkpoints never see its weights.
        self.done_head = DoneHead(self.config["emb_dim"], done_hidden_dim).to(self.device)

        self.checkpoint_path = "checkpoints/smolvlm2.pth"

        summary(self.model, depth=2) 


    def save_the_model(self, checkpoint_path=None):
        if checkpoint_path == None:
            checkpoint_path = self.checkpoint_path

        # The frozen backbone comes back from from_pretrained, so only save what trains.
        trainable = {name: param.detach().cpu()
                     for name, param in self.model.named_parameters()
                     if param.requires_grad}

        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        torch.save({"revision": self.config["revision"], "trainable": trainable}, checkpoint_path)
        print(f"Saved model checkpoint at {checkpoint_path}")


    def load_the_model(self, checkpoint_path=None):
        if checkpoint_path == None:
            checkpoint_path = self.checkpoint_path

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if checkpoint["revision"] != self.config["revision"]:
            raise ValueError(
                f"{checkpoint_path} was trained on revision {checkpoint['revision']}, "
                f"but this model is {self.config['revision']}")

        # strict=False because the frozen params are missing by design.
        result = self.model.load_state_dict(checkpoint["trainable"], strict=False)
        if result.unexpected_keys:
            raise ValueError(
                f"{checkpoint_path} has parameters this model does not: "
                f"{result.unexpected_keys}")
        print(f"Successfully loaded weights from {checkpoint_path}")


    def save_done_head(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.done_head.state_dict(), path)


    def load_done_head(self, path):
        self.done_head.load_state_dict(torch.load(path, map_location=self.device))


    def _add_joint_state(self, prompts, joint_pos):
        """Append joint positions as 0-99 integer bins: ~25 tokens, equal resolution per joint."""
        lo, hi = JOINT_MIN.to(joint_pos.device), JOINT_MAX.to(joint_pos.device)
        bins = ((joint_pos - lo) / (hi - lo) * 99).round().clamp(0, 99).int().tolist()
        return [f"{p}. Joints: {' '.join(map(str, b))}" for p, b in zip(prompts, bins)]


    @torch.no_grad
    def get_action(self, images, prompts, joint_pos):
        logits = self.get_action_logits(images, prompts, joint_pos)
    
        action = torch.clamp(logits, -1, 1)

        return action


    @torch.no_grad
    def get_action_and_done(self, images, prompts, joint_pos):
        """Clamped action chunks and the done probability, from one forward pass."""
        outputs = self._forward(images, prompts, joint_pos)
        action = torch.clamp(self._action_logits(outputs, len(prompts)), -1, 1)
        return action, torch.sigmoid(self._done_logits(outputs))


    def get_action_logits(self, images, prompts, joint_pos):
        outputs = self._forward(images, prompts, joint_pos)
        return self._action_logits(outputs, len(prompts))


    def get_action_and_done_logits(self, images, prompts, joint_pos):
        """Action chunks and done logits for training, from one forward pass."""
        outputs = self._forward(images, prompts, joint_pos)
        return self._action_logits(outputs, len(prompts)), self._done_logits(outputs)


    def get_done_logits(self, images, prompts, joint_pos):
        with torch.no_grad():
            outputs = self._forward(images, prompts, joint_pos)
        return self._done_logits(outputs)


    def _action_logits(self, outputs, batch_size):
        logits = outputs.logits[:, -1, :].float()
        return logits.view(batch_size, self.chunk_size, self.num_actions)


    def _done_logits(self, outputs):
        # detach keeps done's loss from reaching the backbone, so it can't disturb the action weights.
        feature = outputs.hidden_states[-1][:, -1, :].detach()
        return self.done_head(feature).squeeze(-1)


    def _forward(self, images, prompts, joint_pos):
        prompts = self._add_joint_state(prompts, joint_pos)

        text_batch = []

        for prompt in prompts:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                    ],
                }]
            # Non-empty processor_kwargs stops SmolVLM leaking video kwargs into a warning.
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True,
                processor_kwargs={"do_image_splitting": False})
            text_batch.append(text)
        
        image_batch = [[image] for image in images]



        inputs = self.processor(text=text_batch, images=image_batch, return_tensors="pt", padding=True, do_image_splitting=False,
                                do_resize=False)
        inputs = inputs.to(self.device, dtype=self.model.dtype)

        # Runs matmuls in bf16 so the fp32 and bf16 layers can mix; gradients still land on fp32 weights.
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            return self.model(**inputs, output_hidden_states=True)



class FP32Linear(torch.nn.Linear):
    def forward(self, input):
        return super().forward(input.float())


class DoneHead(nn.Module):

    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x))
        return self.output(x)


class RobotControlOutputHead(nn.Module):
    
    def __init__(self,
                 vla_output_dim,
                 hidden_dim=756,
                 n_actions=9
                 ):
        super().__init__()

        self.compression_layer = FP32Linear(in_features=vla_output_dim, 
                                            out_features=hidden_dim,
                                            dtype=torch.float32)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.compression_layer(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output(x)
        return x




