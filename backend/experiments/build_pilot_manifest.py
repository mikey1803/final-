"""Write an empty pilot-manifest skeleton to disk for hand-authoring.

    python experiments/build_pilot_manifest.py --output docs/live_pilot/pilot_manifest.json

Writes placeholder Task records (TODO goal text) -- 4 per agent-side type
(targeted=True) + 8 natural tasks (targeted=False) by default, 28 total,
within the 25-30 pilot-batch range. Task TEXT is never generated here;
fill in each "goal" (and "steps"/"clarify_canned_answer" where relevant)
by hand, then freeze the file (git add + commit) before running
collection with --manifest, per recovery.live.task_manifest's freeze
check.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery.live.pilot_report import build_pilot_skeleton
from recovery.live.task_manifest import save_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=4, help="targeted tasks per agent-side type")
    parser.add_argument("--n-natural", type=int, default=8, help="untargeted/natural tasks")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing file: {args.output}")

    tasks = build_pilot_skeleton(per_type=args.per_type, n_natural=args.n_natural)
    save_manifest(tasks, args.output)
    n_targeted = sum(1 for t in tasks if t.targeted)
    print(f"Wrote {len(tasks)} task(s) to {args.output} "
          f"({n_targeted} targeted, {len(tasks) - n_targeted} natural).")
    print("Fill in each task's goal text by hand, then commit the file before "
          "running collection with --manifest.")


if __name__ == "__main__":
    main()
