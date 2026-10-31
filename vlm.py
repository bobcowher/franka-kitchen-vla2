import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


class VisionLanguageModel:

    def __init__(self, vlm_config, num_actions):

        torch.manual_seed(123)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = vlm_config
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

        self.model.lm_head = FP32Linear(
                in_features=self.config["emb_dim"],
                out_features=num_actions,
                dtype=torch.float32
                )
        
        self.model.to(self.device)

        self.checkpoint_path = "checkpoints/smolvlm2.pth"


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



class FP32Linear(torch.nn.Linear):
    def forward(self, input):
        return super().forward(input.float())
