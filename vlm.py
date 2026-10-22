import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


class VisionLanguageModel:

    def __init__(self, vlm_config):

        torch.manual_seed(123)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = vlm_config
        self.processor = AutoProcessor.from_pretrained(self.config["model_id"])
        self.model = AutoModelForImageTextToText.from_pretrained(
                self.config["model_id"],
                dtype=getattr(torch, self.config["dtype"])
                )
        self.model.to(self.device)

        self.checkpoint_path = "checkpoints/smolvlm2.pth"


    def configure_model(self, num_actions):
        for param in self.model.parameters():
            param.requires_grad = False

        torch.manual_seed(123)

        self.model.lm_head = torch.nn.Linear(
                in_features=self.config["emb_dim"],
                out_features=num_actions,
                dtype=torch.bfloat16
                )

        self.model.to(self.device)


    def save_the_model(self, checkpoint_path=None):
        if checkpoint_path == None:
            checkpoint_path = self.checkpoint_path

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Saved model checkpoint at {checkpoint_path}")


    def load_the_model(self, checkpoint_path=None):
        if checkpoint_path == None:
            checkpoint_path = self.checkpoint_path

        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        print(f"Successfully loaded weights from {checkpoint_path}")


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
