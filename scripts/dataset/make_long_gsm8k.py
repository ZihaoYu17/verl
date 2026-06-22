#!/usr/bin/env python3
"""Create long-prompt GSM8K parquet files for PrefixGrouper experiments.

The output schema matches verl's GSM8K preprocessing script:

    data_source, prompt, ability, reward_model, extra_info

Only the prompt is changed: a deterministic filler context is prepended before
the original GSM8K question. The ground-truth answer is unchanged.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import datasets


FILLER_PARAGRAPH = """
Background note {idx}: This paragraph is intentionally irrelevant to the math
problem. It is included only to create a long shared prompt for systems
profiling. The final question appears after all background notes. Do not use
this paragraph as evidence for the arithmetic answer.
""".strip()

INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def extract_solution(solution_str: str) -> str:
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    if solution is None:
        raise ValueError(f"Could not extract GSM8K answer from: {solution_str[:200]}")
    return solution.group(1).replace(",", "")


def load_tokenizer(tokenizer_path: str | None):
    if not tokenizer_path:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required when --tokenizer_path is set") from exc
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def count_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        # Rough fallback. Use a conservative approximation for English text.
        return max(1, len(text.split()))
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_long_question(
    question: str,
    target_prompt_tokens: int,
    tokenizer,
    min_filler_paragraphs: int = 1,
) -> tuple[str, int, int]:
    """Prepend filler until the user-message content reaches target length."""
    suffix = f"\n\nNow solve the following math problem.\n{question} {INSTRUCTION}"
    filler_parts: list[str] = []
    idx = 1

    while True:
        current = "\n\n".join(filler_parts + [suffix])
        token_count = count_tokens(current, tokenizer)
        if token_count >= target_prompt_tokens and len(filler_parts) >= min_filler_paragraphs:
            return current, token_count, len(filler_parts)
        filler_parts.append(FILLER_PARAGRAPH.format(idx=idx))
        idx += 1


def convert_split(split_dataset, split: str, target_prompt_tokens: int, tokenizer):
    # Keep the original data_source so verl's built-in GSM8K reward function is used.
    data_source = "openai/gsm8k"

    def process_fn(example, idx):
        question_raw = example["question"]
        answer_raw = example["answer"]
        solution = extract_solution(answer_raw)
        long_question, prompt_token_count, filler_paragraphs = build_long_question(
            question=question_raw,
            target_prompt_tokens=target_prompt_tokens,
            tokenizer=tokenizer,
        )
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": long_question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
                "target_prompt_tokens": target_prompt_tokens,
                "measured_prompt_tokens": prompt_token_count,
                "filler_paragraphs": filler_paragraphs,
                "long_prompt_variant": f"gsm8k-long-{target_prompt_tokens}",
            },
        }

    return split_dataset.map(process_fn, with_indices=True, remove_columns=split_dataset.column_names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default=None, help="Optional local raw GSM8K dataset path.")
    parser.add_argument("--tokenizer_path", default=None, help="Tokenizer/model path used to estimate prompt tokens.")
    parser.add_argument(
        "--target_prompt_tokens",
        type=int,
        nargs="+",
        default=[1024],
        help="One or more target prompt lengths, e.g. 1024 2048 4096.",
    )
    parser.add_argument("--local_save_root", default="~/data/long_gsm8k")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_path)

    if args.local_dataset_path:
        dataset = datasets.load_dataset(args.local_dataset_path, "main")
    else:
        dataset = datasets.load_dataset("openai/gsm8k", "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_test_samples is not None:
        test_dataset = test_dataset.select(range(min(args.max_test_samples, len(test_dataset))))

    save_root = Path(os.path.expanduser(args.local_save_root))
    save_root.mkdir(parents=True, exist_ok=True)

    for target in args.target_prompt_tokens:
        out_dir = save_root / f"p{target}"
        out_dir.mkdir(parents=True, exist_ok=True)

        train_out = convert_split(train_dataset, "train", target, tokenizer)
        test_out = convert_split(test_dataset, "test", target, tokenizer)

        train_path = out_dir / "train.parquet"
        test_path = out_dir / "test.parquet"
        train_out.to_parquet(str(train_path))
        test_out.to_parquet(str(test_path))

        train_lengths = train_out["extra_info"]
        measured = [item["measured_prompt_tokens"] for item in train_lengths]
        print(
            f"target={target} saved={out_dir} "
            f"train={len(train_out)} test={len(test_out)} "
            f"prompt_tokens_mean={sum(measured) / len(measured):.1f} "
            f"min={min(measured)} max={max(measured)}"
        )


if __name__ == "__main__":
    main()
