import os
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch.nn as nn
from torchinfo import summary


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

        # Trained weights need fp32 so Adam's small updates aren't rounded away; autocast below bridges the dtypes.
        self.model.model.text_model.layers[-4:].float()
        for param in self.model.model.text_model.layers[-4:].parameters():
            param.requires_grad = True

        self.model.lm_head = RobotControlOutputHead(vla_output_dim=self.config["emb_dim"],
                                                    n_actions=num_actions)
        
        self.model.to(self.device)

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
            outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :].float()

        return logits



class FP32Linear(torch.nn.Linear):
    def forward(self, input):
        return super().forward(input.float())

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




