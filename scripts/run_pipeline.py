"""
run_pipeline.py — Run all Part 1 steps in sequence.

Usage:
    python run_pipeline.py          # run all steps
    python run_pipeline.py --step 2 # run only step 2
    python run_pipeline.py --from 3 # run steps 3, 4, 5+6
"""

import sys
import argparse
import importlib


STEPS = {
    2: ("step2_extract",  "Text Extraction"),
    3: ("step3_clean",    "Text Cleaning"),
    4: ("step4_structure","Document Structuring"),
    5: ("step5_6_chunk",  "Legal Chunking & Retrieval Dataset"),
}


def run_step(step_num: int) -> None:
    module_name, label = STEPS[step_num]
    print(f"\n{'#'*60}")
    print(f"#  STEP {step_num} — {label}")
    print(f"{'#'*60}")
    module = importlib.import_module(module_name)
    module.main()


def main():
    parser = argparse.ArgumentParser(description="Libyan Gov AI — Part 1 Pipeline Runner")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--step", type=int, choices=[2, 3, 4, 5],
                       help="Run only this step")
    group.add_argument("--from", dest="from_step", type=int, choices=[2, 3, 4, 5],
                       help="Run from this step onwards")
    args = parser.parse_args()

    if args.step:
        run_step(args.step)
    elif args.from_step:
        for s in sorted(STEPS.keys()):
            if s >= args.from_step:
                run_step(s)
    else:
        for s in sorted(STEPS.keys()):
            run_step(s)

    print(f"\n\n{'='*60}")
    print("🎉  PART 1 COMPLETE!")
    print("    Dataset is ready at: data/chunks/all_chunks.json")
    print("    Next: Part 2 — Embeddings + Vector DB (ChromaDB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
