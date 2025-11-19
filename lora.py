import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model
from scipy.stats import spearmanr


# -----------------------------
# Dataset
# -----------------------------

class DMSSingleAssayDataset(Dataset):
    """
    Simple dataset for one DMS assay CSV with columns:
      - mutated_sequence
      - DMS_score
    """

    def __init__(self, csv_path, tokenizer, max_length=512):
        df = pd.read_csv(csv_path)
        if "mutated_sequence" not in df.columns or "DMS_score" not in df.columns:
            raise ValueError("CSV must contain 'mutated_sequence' and 'DMS_score' columns.")

        self.seqs = df["mutated_sequence"].astype(str).tolist()
        self.labels = df["DMS_score"].astype("float32").tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        label = self.labels[idx]

        enc = self.tokenizer(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float32)
        return item


# -----------------------------
# Model wrapper: Dayhoff + LoRA + regression head
# -----------------------------

class DayhoffLoRARegression(nn.Module):
    """
    Wraps an AutoModelForCausalLM (Dayhoff) with PEFT LoRA,
    and adds a small regression head on top of pooled hidden states.
    """

    def __init__(self, base_lm, hidden_size):
        super().__init__()
        self.base_lm = base_lm
        self.reg_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        # We just need hidden states; LM loss is irrelevant.
        outputs = self.base_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]  # (B, T, D)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)  # (B, T, 1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts  # mean pooling over non-pad tokens
        else:
            pooled = hidden[:, 0, :]  # fallback

        preds = self.reg_head(pooled).squeeze(-1)  # (B,)

        loss = None
        if labels is not None:
            labels = labels.float()
            loss = F.mse_loss(preds, labels)

        return {"loss": loss, "logits": preds}


# -----------------------------
# Metrics (Spearman + MSE)
# -----------------------------

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.squeeze()
    labels = labels.squeeze()

    # MSE
    mse = float(((preds - labels) ** 2).mean())

    # Spearman
    if np.std(preds) == 0 or np.std(labels) == 0:
        rho = 0.0
    else:
        rho, _ = spearmanr(labels, preds)

    return {"spearman": rho, "mse": mse}


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HF model id, e.g. microsoft/Dayhoff-3b-GR-HM",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to single DMS assay CSV (mutated_sequence, DMS_score)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max token length for sequences",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device train batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for LoRA + head",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------
    # Load tokenizer & base LM
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    base_lm = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    # -------------------------
    # Setup LoRA (PEFT)
    # -------------------------
    # NOTE: target_modules may need to be adjusted to match Dayhoff's attention proj names.
    # Common ones: "q_proj", "k_proj", "v_proj", "o_proj".
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # adjust if needed
    )
    peft_lm = get_peft_model(base_lm, lora_config)
    peft_lm.print_trainable_parameters()

    hidden_size = peft_lm.config.hidden_size
    model = DayhoffLoRARegression(peft_lm, hidden_size).to(device)

    # -------------------------
    # Dataset & splits
    # -------------------------
    full_ds = DMSSingleAssayDataset(args.csv_path, tokenizer, max_length=args.max_length)
    n = len(full_ds)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test])

    print(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # -------------------------
    # Training
    # -------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,

        # Older-style evaluation:
        do_eval=True,
        eval_steps=200,        # pick frequency (validation every 200 steps)
        save_steps=200,        # checkpoint every 200 steps
        save_total_limit=1,    # keep best checkpoint only

        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # -------------------------
    # Final evaluation on val + test
    # -------------------------
    print("\nFinal evaluation on validation set:")
    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    print(val_metrics)

    print("\nEvaluation on test set:")
    test_metrics = trainer.evaluate(eval_dataset=test_ds)
    print(test_metrics)


if __name__ == "__main__":
    main()
