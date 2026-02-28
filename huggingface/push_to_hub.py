#!/usr/bin/env python3
"""
NRC Hub Push Script
====================
Pushes model card, generated dataset, and source to HuggingFace Hub.
Requires:  pip install huggingface_hub
           huggingface-cli login   (or set HF_TOKEN env variable)

Usage:
    python push_to_hub.py --repo-id Nexus-Resonance-Codex/nrc-Ai-Enhancements
    python push_to_hub.py --repo-id Nexus-Resonance-Codex/nrc-Ai-Enhancements --dataset ../huggingface/nrc_dataset.parquet
"""

import argparse
import os
import sys
from pathlib import Path

def push(repo_id: str, model_card_path: str, dataset_path: str | None) -> None:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("[error] huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[info] HF_TOKEN not set in environment. Attempting to use cached login.")
        print("       Run 'huggingface-cli login' if this fails.\n")

    api = HfApi(token=token)

    # ── Create Model Repo ──────────────────────────────────────
    print(f"[1/3] Creating / accessing repo: {repo_id} ...")
    create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        private=False,
        token=token,
    )

    # ── Upload Model Card ──────────────────────────────────────
    print(f"[2/3] Uploading model card ...")
    api.upload_file(
        path_or_fileobj=model_card_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="feat: Add NRC model card",
    )

    # ── Upload Dataset (optional) ──────────────────────────────
    if dataset_path and Path(dataset_path).exists():
        print(f"[3/3] Uploading dataset: {dataset_path} ...")
        api.upload_file(
            path_or_fileobj=dataset_path,
            path_in_repo=f"data/{Path(dataset_path).name}",
            repo_id=repo_id,
            repo_type="model",
            commit_message="feat: Add NRC synthetic training dataset",
        )
    else:
        print(f"[3/3] No dataset provided — skipping dataset upload.")

    print(f"\n✓ Done! View at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Nexus-Resonance-Codex/nrc-Ai-Enhancements")
    parser.add_argument("--card", default=str(Path(__file__).parent / "MODEL_CARD.md"))
    parser.add_argument("--dataset", default=None, help="Path to a generated .parquet dataset file")
    args = parser.parse_args()

    push(args.repo_id, args.card, args.dataset)


if __name__ == "__main__":
    main()
