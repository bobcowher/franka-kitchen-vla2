import torch
import tiktoken
import numpy as np
from models import GPTModel
import os
import matplotlib.pyplot as plt


# TEMP: GPT-2 weight loading helpers
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

class LanguageModel:

    def __init__(self, gpt_config, train_loader=None, val_loader=None):

        torch.manual_seed(123)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = gpt_config
        self.model = GPTModel(cfg=self.config)
        self.optimizer = torch.optim.AdamW(
                         self.model.parameters(),
                         lr=0.0004, weight_decay=0.1
                        )
        self.model.to(self.device)
        self.train_loader = train_loader
        self.val_loader  = val_loader

        self.tokenizer = tiktoken.get_encoding("gpt2")

        self.checkpoint_path = "checkpoints/model.pth"


    def train(self, num_epochs, eval_freq, eval_iter, start_context):

        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1

        for epoch in range(num_epochs):
            self.model.train()

            for input_batch, target_batch in self.train_loader:
                self.optimizer.zero_grad()
                loss = self.calc_loss_batch(input_batch, target_batch)
                loss.backward()
                self.optimizer.step()
                tokens_seen += input_batch.numel()
                global_step += 1

                if global_step % eval_freq == 0:
                    train_loss, val_loss = self.evaluate_model(eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, "
                          f"Val loss {val_loss:.3f}"
                          )

                    self.generate_and_print_sample(start_context)

            self.save_the_model()

        return train_losses, val_losses, track_tokens_seen


    def train_classifier_simple(self, num_epochs, eval_freq, eval_iter):

        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        examples_seen, global_step = 0, -1

        for epoch in range(num_epochs):
            self.model.train()

            for input_batch, target_batch in self.train_loader:
                self.optimizer.zero_grad()
                loss = self.calc_loss_batch(input_batch, target_batch)
                loss.backward()
                self.optimizer.step()
                examples_seen += input_batch.numel()
                global_step += 1

                if global_step % eval_freq == 0:
                    train_loss, val_loss = self.evaluate_model(eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, "
                          f"Val loss {val_loss:.3f}"
                          )

                    self.save_the_model()

                    train_accuracy = self.calc_accuracy_loader(
                            self.train_loader, num_batches=eval_iter
                            )

                    val_accuracy = self.calc_accuracy_loader(
                            self.val_loader, num_batches=eval_iter
                            )

                    print(f"Train accuracy at epoch {epoch}: {train_accuracy:.3f}")
                    print(f"Val accuracy at epoch {epoch}: {val_accuracy:.3f}")
                    
                    train_accs.append(train_accuracy)
                    val_accs.append(val_accuracy)

        self.save_the_model(checkpoint_path="checkpoints/review_classifier.pth")

        return train_losses, val_losses, train_accs, val_accs, examples_seen 





    def fine_tune(self):
        for param in self.model.parameters():
            param.requires_grad = False

        torch.manual_seed(123)

        num_classes = 2

        self.model.lm_head = torch.nn.Linear(
                in_features=self.config["emb_dim"],
                out_features=num_classes
                )

        for param in self.model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in self.model.final_norm.parameters():
            param.requires_grad = True

        
        
    def save_the_model(self, checkpoint_path=None):
        if checkpoint_path == None:
            checkpoint_path = self.checkpoint_path

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Saved model checkpoint at {checkpoint_path}")


    def load_the_model(self, checkpoint_path=None):
        if checkpoint_path == None:
            checkpoint_path = self.checkpoint_path

        try:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"Successfully loaded weights from {checkpoint_path}")
        except:
            print(f"Failed to load weights from {checkpoint_path}")


    def generate_text_simple(self, idx,
                             max_new_tokens, context_size):

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = self.model(idx_cond)

            logits = logits[:, -1, :] # grabs the last time step
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


    def generate(self, idx, max_new_tokens, context_size,
                 temperature=0.0, top_k=None, eos_id=None):

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = self.model(idx_cond)

            logits = logits[:, -1, :] # grabs the last time step

            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                        logits < min_val,
                        torch.tensor(float('-inf')).to(logits.device),
                        logits
                        )
            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            if idx_next == eos_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def classify_review(self, text, max_length=None, pad_token_id=50256):
        self.model.eval()

        input_ids = self.tokenizer.encode(text)
        supported_context_length = self.model.pos_emb.weight.shape[0]

        input_ids = input_ids[:min(max_length, supported_context_length)]

        input_ids += [pad_token_id] * (max_length - len(input_ids))

        input_tensor = torch.tensor(input_ids, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(input_tensor)[:, -1, :]

        print(logits)

        predicted_label = torch.argmax(logits, dim=-1).item()
        print(predicted_label)
        return "spam" if predicted_label == 1 else "not spam"



    def text_to_token_ids(self, text, tokenizer):
        encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        return encoded_tensor


    def token_ids_to_text(self, token_ids, tokenizer):
        flat = token_ids.squeeze(0)
        return tokenizer.decode(flat.tolist())


    def calc_loss_batch(self, input_batch, target_batch):
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)

        logits = self.model(input_batch)[:, -1, :]
        loss = torch.nn.functional.cross_entropy(logits, target_batch)
        return loss


    def calc_loss_loader(self, data_loader, num_batches=None):
        total_loss = 0.

        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = self.calc_loss_batch(input_batch, target_batch)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches


    def calc_accuracy_loader(self, data_loader, num_batches=None):
        self.model.eval()
        correct_predictions, num_exampels = 0, 0

        if num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                with torch.no_grad():
                    logits = self.model(input_batch)[:, -1, :] # Logits of last output token.
                predicted_labels = torch.argmax(logits, dim=-1)

                num_exampels += predicted_labels.shape[0]
                correct_predictions += (
                        (predicted_labels == target_batch).sum().item()
                        )
            else:
                break

        return correct_predictions / num_exampels


    def evaluate_model(self, eval_iter):
        self.model.eval()

        with torch.no_grad():
            train_loss = self.calc_loss_loader(data_loader=self.train_loader, num_batches=eval_iter)
            val_loss = self.calc_loss_loader(data_loader=self.val_loader, num_batches=eval_iter)
            self.model.train()
            return train_loss, val_loss

    def generate_and_print_sample(self, start_context):
        self.model.eval()
        context_size = self.model.pos_emb.weight.shape[0]
        encoded = self.text_to_token_ids(start_context, self.tokenizer).to(self.device)
        with torch.no_grad():
            token_ids = self.generate_text_simple(
                    idx=encoded,
                    max_new_tokens=50, context_size=context_size
                    )
        decoded_text = self.token_ids_to_text(token_ids, self.tokenizer)
        print(decoded_text.replace("\n", " "))
        self.model.train()

    def plot_values(self,
                    epochs_seen,
                    examples_seen,
                    train_values,
                    val_values,
                    label="loss"):
        fig, ax1 = plt.subplots(figsize=(5,3))

        ax1.plot(epochs_seen, train_values, label=f"Training {label}")
        ax1.plot(
                epochs_seen, val_values, linestyle="-.",
                label=f"Validation {label}"
                )
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel(label.capitalize())
        ax1.legend()

        ax2 = ax1.twiny()
        ax2.plot(examples_seen, train_values, alpha=0)
        ax2.set_xlabel("Examples seen")

        fig.tight_layout()
        plt.savefig(f"{label}-plot.pdf")
        # plt.show()


    # TEMP: load pretrained OpenAI GPT-2 weights from gpt_download.download_and_load_gpt2
    def load_gpt_weights(self, params):
        gpt = self.model

        gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
        gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

        for b in range(len(params["blocks"])):
            q_w, k_w, v_w = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.weight = assign(
                gpt.trf_blocks[b].att.W_query.weight, q_w.T)
            gpt.trf_blocks[b].att.W_key.weight = assign(
                gpt.trf_blocks[b].att.W_key.weight, k_w.T)
            gpt.trf_blocks[b].att.W_value.weight = assign(
                gpt.trf_blocks[b].att.W_value.weight, v_w.T)

            q_b, k_b, v_b = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.bias = assign(
                gpt.trf_blocks[b].att.W_query.bias, q_b)
            gpt.trf_blocks[b].att.W_key.bias = assign(
                gpt.trf_blocks[b].att.W_key.bias, k_b)
            gpt.trf_blocks[b].att.W_value.bias = assign(
                gpt.trf_blocks[b].att.W_value.bias, v_b)

            gpt.trf_blocks[b].att.out_proj.weight = assign(
                gpt.trf_blocks[b].att.out_proj.weight,
                params["blocks"][b]["attn"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].att.out_proj.bias = assign(
                gpt.trf_blocks[b].att.out_proj.bias,
                params["blocks"][b]["attn"]["c_proj"]["b"])

            gpt.trf_blocks[b].ff.layers[0].weight = assign(
                gpt.trf_blocks[b].ff.layers[0].weight,
                params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            gpt.trf_blocks[b].ff.layers[0].bias = assign(
                gpt.trf_blocks[b].ff.layers[0].bias,
                params["blocks"][b]["mlp"]["c_fc"]["b"])
            gpt.trf_blocks[b].ff.layers[2].weight = assign(
                gpt.trf_blocks[b].ff.layers[2].weight,
                params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].ff.layers[2].bias = assign(
                gpt.trf_blocks[b].ff.layers[2].bias,
                params["blocks"][b]["mlp"]["c_proj"]["b"])

            gpt.trf_blocks[b].norm1.scale = assign(
                gpt.trf_blocks[b].norm1.scale,
                params["blocks"][b]["ln_1"]["g"])
            gpt.trf_blocks[b].norm1.shift = assign(
                gpt.trf_blocks[b].norm1.shift,
                params["blocks"][b]["ln_1"]["b"])
            gpt.trf_blocks[b].norm2.scale = assign(
                gpt.trf_blocks[b].norm2.scale,
                params["blocks"][b]["ln_2"]["g"])
            gpt.trf_blocks[b].norm2.shift = assign(
                gpt.trf_blocks[b].norm2.shift,
                params["blocks"][b]["ln_2"]["b"])

        gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
        gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
        gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

        gpt.to(self.device)
        print("Loaded pretrained GPT-2 weights")
