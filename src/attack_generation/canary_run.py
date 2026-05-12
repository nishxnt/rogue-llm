"""Canary pipeline: mutate LLM01 seeds and run all 4 quality filter stages.

Usage:
    python -m src.attack_generation.canary_run

Reads  : attacks/seeds/LLM01_prompt_injection.yaml
Writes : attacks/v1/checkpoints/LLM01.jsonl   (survivors)
         attacks/v1/rejected.jsonl             (filtered-out variants + reason)
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from src.attack_generation.mutator import mutate_seed
from src.attack_generation.quality_filter import (
    filter_dedup,
    filter_length,
    filter_llm_validator,
    filter_metadata,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

SEED_FILE = Path("attacks/seeds/LLM01_prompt_injection.yaml")
CHECKPOINT_FILE = Path("attacks/v1/checkpoints/LLM01.jsonl")
REJECTED_FILE = Path("attacks/v1/rejected.jsonl")
VARIANTS_PER_SEED = 3
EXPECTED_CATEGORY = "LLM01:2025"


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def main() -> None:
    # ── Load seeds ────────────────────────────────────────────────────────────
    with SEED_FILE.open(encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    seeds: list[dict[str, Any]] = doc["seeds"]
    logger.info("Loaded %d seeds from %s", len(seeds), SEED_FILE)

    # ── Mutation ──────────────────────────────────────────────────────────────
    all_variants: list[dict[str, Any]] = []
    for seed in seeds:
        logger.info("Mutating seed %s (%s)…", seed["id"], seed["attack_strategy"])
        variants = mutate_seed(seed, n_variants=VARIANTS_PER_SEED)
        logger.info("  → got %d variants", len(variants))
        all_variants.extend(variants)

    generated_count = len(all_variants)
    logger.info("Total generated: %d", generated_count)

    # ── Filter stage 1: length ────────────────────────────────────────────────
    survivors, rejected_length = filter_length(all_variants)
    logger.info("Stage 1 (length): %d survived / %d rejected", len(survivors), len(rejected_length))

    # ── Filter stage 2: metadata regex ────────────────────────────────────────
    survivors, rejected_meta = filter_metadata(survivors, expected_category=EXPECTED_CATEGORY)
    logger.info("Stage 2 (metadata): %d survived / %d rejected", len(survivors), len(rejected_meta))

    # ── Filter stage 3: MinHash dedup ────────────────────────────────────────
    survivors, rejected_dedup = filter_dedup(survivors)
    logger.info("Stage 3 (dedup): %d survived / %d rejected", len(survivors), len(rejected_dedup))

    # ── Filter stage 4: LLM category validator ────────────────────────────────
    survivors, rejected_llm = filter_llm_validator(survivors, expected_category=EXPECTED_CATEGORY)
    logger.info(
        "Stage 4 (llm_validator): %d survived / %d rejected", len(survivors), len(rejected_llm)
    )

    # ── Assign final IDs ──────────────────────────────────────────────────────
    for i, v in enumerate(survivors, start=1):
        v["id"] = f"LLM01-{i:04d}"

    # ── Persist ───────────────────────────────────────────────────────────────
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text("")  # truncate / create
    REJECTED_FILE.write_text("")

    _append_jsonl(CHECKPOINT_FILE, survivors)
    all_rejected = rejected_length + rejected_meta + rejected_dedup + rejected_llm
    _append_jsonl(REJECTED_FILE, all_rejected)

    logger.info("Wrote %d survivors to %s", len(survivors), CHECKPOINT_FILE)
    logger.info("Wrote %d rejected to %s", len(all_rejected), REJECTED_FILE)

    # ── Summary report ────────────────────────────────────────────────────────
    _print_report(
        seeds=seeds,
        generated=generated_count,
        after_length=generated_count - len(rejected_length),
        after_meta=generated_count - len(rejected_length) - len(rejected_meta),
        after_dedup=generated_count
        - len(rejected_length)
        - len(rejected_meta)
        - len(rejected_dedup),
        final=len(survivors),
        survivors=survivors,
        all_rejected=all_rejected,
        rejected_llm=rejected_llm,
    )


def _print_report(
    seeds: list[dict[str, Any]],
    generated: int,
    after_length: int,
    after_meta: int,
    after_dedup: int,
    final: int,
    survivors: list[dict[str, Any]],
    all_rejected: list[dict[str, Any]],
    rejected_llm: list[dict[str, Any]],
) -> None:
    sep = "=" * 72

    print(f"\n{sep}")
    print("CANARY RUN REPORT — LLM01:2025 Prompt Injection")
    print(sep)

    print(f"\n{'Variants generated (pre-filter)':<40}: {generated}")
    print(f"{'After Stage 1 — length filter':<40}: {after_length}")
    print(f"{'After Stage 2 — metadata regex':<40}: {after_meta}")
    print(f"{'After Stage 3 — MinHash dedup (0.85)':<40}: {after_dedup}")
    print(f"{'After Stage 4 — LLM validator (final)':<40}: {final}")

    # Per-strategy survival
    print(f"\n{'Per-strategy survival':}")
    print(f"  {'Strategy':<35} {'Survived':>8} {'of':>4} {VARIANTS_PER_SEED:>4}")
    strategy_map: dict[str, list[str]] = {}
    for s in seeds:
        strategy_map[s["id"]] = s["attack_strategy"]

    strategy_counts: dict[str, dict[str, int]] = {}
    for s in seeds:
        strategy_counts[s["attack_strategy"]] = {"survived": 0, "total": VARIANTS_PER_SEED}

    for v in survivors:
        strat = strategy_map.get(v["variant_of"], v.get("attack_strategy", "unknown"))
        if strat in strategy_counts:
            strategy_counts[strat]["survived"] += 1

    for strat, counts in strategy_counts.items():
        bar = "✓" * counts["survived"] + "✗" * (counts["total"] - counts["survived"])
        print(f"  {strat:<35} {counts['survived']:>8}/{counts['total']}  {bar}")

    # 3 example survivors
    print(f"\n{'Example surviving variants (up to 3)':}")
    shown_strategies: set[str] = set()
    shown = 0
    for v in survivors:
        if shown >= 3:
            break
        strat = v.get("attack_strategy", "")
        if strat in shown_strategies:
            continue
        shown_strategies.add(strat)
        shown += 1
        _print_variant_block(v, label=f"Survivor {shown} [{strat}]")

    # If we didn't get 3 with unique strategies, just show whatever's left
    for v in survivors:
        if shown >= 3:
            break
        if v.get("attack_strategy") not in shown_strategies:
            shown += 1
            _print_variant_block(v, label=f"Survivor {shown} [{v.get('attack_strategy')}]")

    # 3 example rejected variants
    print(f"\n{'Example rejected variants (up to 3)':}")
    for i, v in enumerate(all_rejected[:3], start=1):
        _print_rejected_block(
            v,
            label=f"Rejected {i} [stage={v.get('rejection_stage', '?')}]",
        )

    # Flag if any single stage drops >= 50%
    drops = [
        ("length", generated, after_length),
        ("metadata_regex", after_length, after_meta),
        ("minhash_dedup", after_meta, after_dedup),
        ("llm_validator", after_dedup, final),
    ]
    flags = []
    for stage_name, before, after in drops:
        if before > 0 and (before - after) / before >= 0.5:
            flags.append(
                f"  ⚠  Stage '{stage_name}' dropped {before - after}/{before} "
                f"({(before - after) / before * 100:.0f}%) — mutator prompt may need tuning"
            )
    if flags:
        print(f"\n{'FLAGS':}")
        for f in flags:
            print(f)
    else:
        print("\nNo filter stage dropped ≥50% of variants.")

    print(f"\n{sep}\n")


def _print_variant_block(v: dict[str, Any], label: str) -> None:
    print(f"\n  [{label}]")
    print(f"  variant_of : {v.get('variant_of')}")
    print(f"  strategy   : {v.get('attack_strategy')}")
    print(f"  tokens     : {v.get('token_count', 'n/a')}")
    prompt = v["prompt_text"]
    preview = (prompt[:300] + "…") if len(prompt) > 300 else prompt
    print(f"  prompt     :\n    {preview.replace(chr(10), chr(10) + '    ')}")


def _print_rejected_block(v: dict[str, Any], label: str) -> None:
    print(f"\n  [{label}]")
    print(f"  variant_of : {v.get('variant_of')}")
    print(f"  strategy   : {v.get('attack_strategy')}")
    print(f"  reason     : {v.get('rejection_reason', '?')}")
    prompt = v.get("prompt_text", "")
    preview = (prompt[:200] + "…") if len(prompt) > 200 else prompt
    print(f"  prompt     :\n    {preview.replace(chr(10), chr(10) + '    ')}")


if __name__ == "__main__":
    main()
