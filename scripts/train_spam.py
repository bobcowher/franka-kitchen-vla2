import sys
import torch
from pathlib import Path
import tiktoken
from torch.utils.data import DataLoader

# Make the project root importable when run as scripts/<name>.py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import calc_accuracy_loader

from languagemodel import LanguageModel
from dataset import SpamDataset

tokenizer = tiktoken.get_encoding("gpt2")

train_dataset = SpamDataset(
        csv_file="data/train.csv",
        max_length=None,
        tokenizer=tokenizer)

val_dataset = SpamDataset(
        csv_file="data/validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
        )
test_dataset = SpamDataset(
        csv_file="data/test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
        )

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
        )

val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False
        )

test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False
        )

# for input_batch, target_batch in train_loader:
#
#     print("Input batch dimensions:", input_batch.shape)
#     print("Label batch dimensions", target_batch.shape)

# print(f"{len(train_loader)} training batches")
# print(f"{len(val_loader)} validation batches")
# print(f"{len(test_loader)} test batches")

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
        }

language_model_configs = {
"gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
"gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
"gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
"gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(language_model_configs[CHOOSE_MODEL])

language_model = LanguageModel(gpt_config=BASE_CONFIG, 
                      train_loader=train_loader, 
                      val_loader=val_loader
                      )


text_1 = "Every effort moves you"

# language_model.generate_and_print_sample(start_context=text_1)

text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
    )

language_model.load_the_model("checkpoints/gpt.pth")

language_model.fine_tune()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
language_model.model.to(device)

torch.manual_seed(123)

train_accuracy = calc_accuracy_loader(
        train_loader, language_model.model, device, num_batches=10
        )

test_accuracy = calc_accuracy_loader(
        test_loader, language_model.model, device, num_batches=10
        )

val_accuracy = calc_accuracy_loader(
        val_loader, language_model.model, device, num_batches=10  
        )

print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Val Accuracy: {val_accuracy}")

with torch.no_grad():
    train_loss = language_model.calc_loss_loader(train_loader, num_batches=5)
    test_loss  = language_model.calc_loss_loader(test_loader, num_batches=5)
    val_loss   = language_model.calc_loss_loader(val_loader, num_batches=5)

print(f"Training loss: {train_loss:.3f}")
print(f"Test loss: {test_loss:.3f}")
print(f"Val loss: {val_loss:.3f}")

import time

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(language_model.model.parameters(),
                              lr=5e-5,
                              weight_decay=0.1)
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen = \
        language_model.train_classifier_simple(
                num_epochs=num_epochs, eval_freq=50, eval_iter=5 
                )

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

print("train_accs", train_accs)
print("val_accs", val_accs)
print("epochs_tensor", epochs_tensor)
print("examples_seen_tensor", examples_seen_tensor)

language_model.plot_values(epochs_seen=epochs_tensor,
                           examples_seen=examples_seen_tensor,
                           train_values=train_losses,
                           val_values=val_losses)

language_model.plot_values(epochs_seen=epochs_tensor,
                           examples_seen=examples_seen_tensor,
                           train_values=train_accs,
                           val_values=val_accs,
                           label="accuracy")
text_1 = (
"You are a winner you have been specially"
" selected to receive $1000 cash or a $2000 award."
)
print(language_model.classify_review(text_1, max_length=train_dataset.max_length))

text_2 = (
"Hey, just wanted to check if we're still on"
" for dinner tonight? Let me know!"
)
print(language_model.classify_review(text_2, max_length=train_dataset.max_length))

    # def plot_values(self,
    #                 epochs_seen,
    #                 examples_seen,
    #                 train_values,
    #                 val_values,
    #                 label="loss"):






# language_model.generate_and_print_sample(start_context=text_2)



# token_ids = language_model.generate_text_simple(idx=model.text_to_token_ids(text_1, tokenizer),
#                                        max_new_tokens=15,
#                                        context_size=BASE_CONFIG["context_length"])
#

# language_model.train(num_epochs=num_epochs,
#             eval_freq=5,
#             eval_iter=5,
#             start_context="Every effort moves you")





    # def plot_values(self,
    #                 epochs_seen,
    #                 examples_seen,
    #                 train_values,
    #                 val_values,
    #                 label="loss"):






# language_model.generate_and_print_sample(start_context=text_2)



# token_ids = language_model.generate_text_simple(idx=model.text_to_token_ids(text_1, tokenizer),
#                                        max_new_tokens=15,
#                                        context_size=BASE_CONFIG["context_length"])
#

# language_model.train(num_epochs=num_epochs,
#             eval_freq=5,
#             eval_iter=5,
#             start_context="Every effort moves you")



