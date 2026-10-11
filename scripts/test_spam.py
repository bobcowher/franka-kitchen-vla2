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

language_model.fine_tune()

language_model.load_the_model(checkpoint_path="checkpoints/review_classifier.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
language_model.model.to(device)

torch.manual_seed(123)

text_1 = (
"You are a winner you have been specially"
" selected to receive $1000 cash or a $2000 award."
)
print(language_model.classify_review(text_1, max_length=train_dataset.max_length))

text_2 = (
"Sounds like a great idea!"
)
print(language_model.classify_review(text_2, max_length=train_dataset.max_length))
