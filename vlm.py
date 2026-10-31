import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


class VisionLanguageModel:

    def __init__(self, vlm_config, num_actions):

        torch.manual_seed(123)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = vlm_config
        self.processor = AutoProcessor.from_pretrained(self.config["model_id"])

        self.processor.tokenizer.padding_side = "left"

        self.model = AutoModelForImageTextToText.from_pretrained(
                self.config["model_id"],
                dtype=getattr(torch, self.config["dtype"])
                )
        
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.lm_head = torch.nn.Linear(
                in_features=self.config["emb_dim"],
                out_features=num_actions,
                dtype=torch.bfloat16
                )
        
        self.model.to(self.device)

        self.checkpoint_path = "checkpoints/smolvlm2.pth"


    def save_the_model(self, checkpoint_path=None):
        if checkpoint_path == None:
            checkpoint_path = self.checkpoint_path

        # Only what trains. The frozen backbone is identical to the pretrained
        # weights from_pretrained loads, so saving it would be ~1 GB of copies.
        # Selected by requires_grad, like the optimizer, so unfrozen layers or
        # LoRA adapters land in the checkpoint without editing this.
        trainable = {name: param.detach().cpu()
                     for name, param in self.model.named_parameters()
                     if param.requires_grad}

        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        torch.save({"trainable": trainable}, checkpoint_path)
        print(f"Saved model checkpoint at {checkpoint_path}")


    def load_the_model(self, checkpoint_path=None):
        if checkpoint_path == None:
            checkpoint_path = self.checkpoint_path

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # strict=False because every frozen param is legitimately missing. A key
        # the model does not have means the checkpoint is from a different head
        # or adapter setup, so that still fails.
        result = self.model.load_state_dict(checkpoint["trainable"], strict=False)
        if result.unexpected_keys:
            raise ValueError(
                f"{checkpoint_path} has parameters this model does not: "
                f"{result.unexpected_keys}")
        print(f"Successfully loaded weights from {checkpoint_path}")


    @torch.no_grad
    def get_action(self, images, prompts):
        logits = self.get_action_logits(images, prompts)
    
        action = torch.clamp(logits, -1, 1)

        return action


    def get_action_logits(self, images, prompts):
    
        text_batch = []

        for prompt in prompts:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                    ],
                }]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            text_batch.append(text)
        
        image_batch = [[image] for image in images]



        inputs = self.processor(text=text_batch, images=image_batch, return_tensors="pt", padding=True, do_image_splitting=False)
        inputs = inputs.to(self.device, dtype=self.model.dtype)

        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]

        return logits 


    def generate(self, image, prompt, max_new_tokens=64):
        self.model.eval()

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
                ],
            }]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="pt")
        inputs = inputs.to(self.device, dtype=self.model.dtype)

        with torch.no_grad():
            token_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        new_tokens = token_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
