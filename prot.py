import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# --------------------
# Scoring utilities
# --------------------

@torch.no_grad()
def mean_logprob_batch(seqs, model, tokenizer, device, max_length=None):
    """
    Compute mean per-token log-prob for each sequence in `seqs`.

    This is analogous to -ce_loss in the Dayhoff training script:
    ce_loss = -mean_logprob, so fw_score = -ce_loss = mean_logprob.

    Returns: np.array of shape [B]
    """
    if len(seqs) == 0:
        return np.array([])

    toks = tokenizer(
        list(seqs),
        return_tensors="pt",
        padding=True,
        truncation=False if max_length is None else True,
        max_length=max_length,
        add_special_tokens=False,
    ).to(device)

    input_ids = toks["input_ids"]          # [B, T]
    attn_mask = toks["attention_mask"]     # [B, T]

    out = model(**toks)
    logits = out.logits                    # [B, T, V]
    logprobs = F.log_softmax(logits, dim=-1)

    # Next-token prediction: compare logits[:, :-1] to tokens[:, 1:]
    next_tokens = input_ids[:, 1:]                    # [B, T-1]
    logp_next = logprobs[:, :-1, :].gather(
        2, next_tokens.unsqueeze(-1)
    ).squeeze(-1)                                      # [B, T-1]

    # Valid positions: where the *next* token is not padding
    valid_mask = attn_mask[:, 1:].bool()              # [B, T-1]
    logp_next = logp_next * valid_mask

    # Sum log-probs over valid positions
    seq_loglik = logp_next.sum(dim=1)                 # [B]
    lengths = valid_mask.sum(dim=1)                   # [B]
    lengths = lengths.clamp(min=1)                    # avoid div by zero

    mean_loglik = seq_loglik / lengths                # [B]
    return mean_loglik.cpu().numpy()


def score_sequences_batched(seqs, model, tokenizer, device, batch_size=8, max_length=None):
    """
    Compute forward, backward, and averaged scores for a list of sequences.
    Returns three np.arrays of shape [N]: fw_scores, bw_scores, seq_scores.
    """
    fw_scores = []
    bw_scores = []

    for start in range(0, len(seqs), batch_size):
        batch_seqs = seqs[start:start + batch_size]

        # Forward score: mean log-prob on original sequences
        fw = mean_logprob_batch(batch_seqs, model, tokenizer, device, max_length=max_length)
        fw_scores.append(fw)

        # Backward score: mean log-prob on reversed sequences
        rev_batch = [s[::-1] for s in batch_seqs]
        bw = mean_logprob_batch(rev_batch, model, tokenizer, device, max_length=max_length)
        bw_scores.append(bw)

    fw_scores = np.concatenate(fw_scores) if fw_scores else np.array([])
    bw_scores = np.concatenate(bw_scores) if bw_scores else np.array([])
    seq_scores = 0.5 * (fw_scores + bw_scores)

    return fw_scores, bw_scores, seq_scores


# --------------------
# Per-assay evaluation
# --------------------

def eval_one_dir(
    dms_dir: Path,
    out_dir: Path,
    model,
    tokenizer,
    device,
    model_name: str,
    batch_size: int = 8,
    max_length: int | None = None,
    label: str = "substitutions",
):
    """
    Evaluate all CSVs in `dms_dir`.

    For each assay:
      - load mutated_sequence + DMS_score
      - compute fw/bw/seq scores
      - compute Spearman per assay
      - save per-assay CSV with scores

    Returns: list of dicts with assay name + spearmans.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    assays = sorted(dms_dir.glob("*.csv"))

    results = []

    print(f"\n=== Evaluating {label} in {dms_dir} ({len(assays)} assays) ===")
    for csv_path in tqdm(assays):
        assay_name = csv_path.stem

        df_in = pd.read_csv(csv_path)
        if "mutated_sequence" not in df_in.columns or "DMS_score" not in df_in.columns:
            print(f"Skipping {assay_name}: missing 'mutated_sequence' or 'DMS_score' column.")
            continue

        df_in = df_in.head(1000)

        seqs = df_in["mutated_sequence"].astype(str).tolist()
        dms_scores = df_in["DMS_score"].to_numpy(dtype=float)

        fw_scores, bw_scores, seq_scores = score_sequences_batched(
            seqs,
            model,
            tokenizer,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )

        # Build output dataframe
        df_out = pd.DataFrame()
        if "mutant" in df_in.columns:
            df_out["mutant"] = df_in["mutant"]
        df_out["mutated_sequence"] = df_in["mutated_sequence"]
        df_out["DMS_score"] = dms_scores
        df_out["assay"] = assay_name

        fw_col = f"{model_name}_fw_score"
        bw_col = f"{model_name}_bw_score"
        seq_col = f"{model_name}_seq_score"

        df_out[fw_col] = fw_scores
        df_out[bw_col] = bw_scores
        df_out[seq_col] = seq_scores

        # Spearman per assay
        fw_spear = spearmanr(fw_scores, dms_scores).statistic
        bw_spear = spearmanr(bw_scores, dms_scores).statistic
        seq_spear = spearmanr(seq_scores, dms_scores).statistic

        print(f"{assay_name}: fw={fw_spear:.4f}, bw={bw_spear:.4f}, seq={seq_spear:.4f}")

        results.append(
            {
                "assay": assay_name,
                "fw_spearman": fw_spear,
                "bw_spearman": bw_spear,
                "seq_spearman": seq_spear,
            }
        )

        out_file = out_dir / f"{model_name}_{assay_name}.csv"
        df_out.to_csv(out_file, index=False)

    return results


# --------------------
# Main
# --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True,
                        help="HF model id, e.g. microsoft/Dayhoff-3b-GR-HM")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root of ProteinGym data with DMS_ProteinGym_* folders")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for per-assay CSVs and summary")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for scoring sequences")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Optional max_length for tokenization (for very long sequences)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    subst_dir = data_root / "DMS_ProteinGym_substitutions"
    # indel_dir = data_root / "DMS_ProteinGym_indels"

    if not subst_dir.exists():
        print(f"Warning: substitutions dir not found: {subst_dir}")
    # if not indel_dir.exists():
    #     print(f"Warning: indels dir not found: {indel_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model {args.model_id} on {device}...")
    cfg = AutoConfig.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=cfg,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device).eval()
    print("✓ Model loaded.")

    # A short name for columns/output
    model_name = args.model_id.split("/")[-1]

    # Substitutions
    subst_out_dir = out_root / "substitutions"
    subst_results = []
    if subst_dir.exists():
        subst_results = eval_one_dir(
            subst_dir,
            subst_out_dir,
            model,
            tokenizer,
            device,
            model_name=model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
            label="substitutions",
        )
        summary_subst = out_root / f"{model_name}_substitutions_summary.csv"
        if subst_results:
            pd.DataFrame(subst_results).to_csv(summary_subst, index=False)
            print(f"\n✓ Wrote substitutions summary to {summary_subst}")

    # Indels (treated the same way: score full mutated_sequence)
    # indel_out_dir = out_root / "indels"
    # indel_results = []
    # if indel_dir.exists():
    #     indel_results = eval_one_dir(
    #         indel_dir,
    #         indel_out_dir,
    #         model,
    #         tokenizer,
    #         device,
    #         model_name=model_name,
    #         batch_size=args.batch_size,
    #         max_length=args.max_length,
    #         label="indels",
    #     )
    #     summary_indels = out_root / f"{model_name}_indels_summary.csv"
    #     if indel_results:
    #         pd.DataFrame(indel_results).to_csv(summary_indels, index=False)
    #         print(f"\n✓ Wrote indels summary to {summary_indels}")

    print("\nDone.")


if __name__ == "__main__":
    main()
