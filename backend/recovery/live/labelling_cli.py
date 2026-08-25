"""Fast terminal labelling tool for live-pilot events.

Walks unlabelled events in an ``EventStore``, one at a time, shows the raw
error and enough context to judge it, and takes a single keystroke for the
failure type, followed by an optional confidence flag and second-choice
type. Writes back immediately (one event = one file = resumable by
construction, same as everything else in this store).

A second mode, ``--relabel``, samples N already-labelled events and
re-presents them blind (no prior label shown) for a second, independent
pass -- stored separately in ``relabel_type``, never overwriting
``true_type`` -- so ``recovery.live.agreement.compute_relabel_kappa`` can
measure self-consistency.

Run directly:
    python -m recovery.live.labelling_cli <store_root>
    python -m recovery.live.labelling_cli <store_root> --relabel [--n 50] [--seed 0]
"""

from __future__ import annotations

import argparse
import getpass
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.failures.taxonomy import FailureType

from recovery.live.agreement import compute_relabel_kappa
from recovery.live.event_store import EventStore

TYPE_KEYS: dict[str, FailureType] = {
    "1": FailureType.TOOL_TIMEOUT,
    "2": FailureType.TOOL_ERROR,
    "3": FailureType.WRONG_TOOL_SELECTED,
    "4": FailureType.HALLUCINATED_OUTPUT,
    "5": FailureType.MALFORMED_OUTPUT,
    "6": FailureType.CONTEXT_LOSS,
    "7": FailureType.AMBIGUOUS_INSTRUCTION,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _print_menu() -> None:
    print("  " + "  ".join(f"[{k}] {v.value}" for k, v in TYPE_KEYS.items()))
    print("  [s] skip for now   [q] quit")


def _print_event(event) -> None:
    print("\n" + "=" * 72)
    print(f"event_id={event.event_id}  session={event.session_id}  step={event.step_index}")
    print(f"tool={event.tool_name}  http_status={event.http_status}  "
          f"injected={event.injected} ({event.injected_kind})")
    print(f"goal: {event.agent_state.goal}")
    print("-" * 72)
    print(f"raw_error:\n{event.raw_error}")
    if event.raw_response_body:
        body_preview = event.raw_response_body[:300]
        print(f"\nraw_response_body (truncated):\n{body_preview}")
    print("=" * 72)


def _prompt_type(prompt_prefix: str = "") -> Optional[str]:
    """Prompt for a single type-key choice. Returns None for skip/quit
    signals ('s'/'q'), which the caller must handle -- this helper only
    loops on invalid keys."""
    while True:
        choice = input(f"{prompt_prefix}> ").strip().lower()
        if choice in ("q", "s"):
            return choice
        if choice in TYPE_KEYS:
            return choice
        print("  not a valid choice, try again")


def _prompt_confidence_and_second_choice(primary_key: str) -> tuple[Optional[str], Optional[FailureType]]:
    """Two quick optional follow-ups after the primary label. Enter alone
    on either defaults to 'confident' / no second choice."""
    conf_raw = input("  confidence [c=confident (default), u=unsure]: ").strip().lower()
    confidence = "unsure" if conf_raw == "u" else "confident"

    second_raw = input("  second choice [type key, Enter to skip]: ").strip().lower()
    second_choice_type: Optional[FailureType] = None
    if second_raw and second_raw != primary_key and second_raw in TYPE_KEYS:
        second_choice_type = TYPE_KEYS[second_raw]
    return confidence, second_choice_type


def label_one(store: EventStore, event, labelled_by: str) -> str:
    """Show one event, read the primary type plus optional confidence/
    second-choice, apply or skip. Returns 'labelled', 'skipped', or 'quit'."""
    _print_event(event)
    _print_menu()

    choice = _prompt_type()
    if choice == "q":
        return "quit"
    if choice == "s":
        return "skipped"

    confidence, second_choice_type = _prompt_confidence_and_second_choice(choice)
    event.label(
        TYPE_KEYS[choice], labelled_by=labelled_by, labelled_at=_now_iso(),
        second_choice_type=second_choice_type, confidence=confidence,
    )
    store.write(event)
    return "labelled"


def relabel_one(store: EventStore, event, labelled_by: str) -> str:
    """Blind second pass: same display as ``label_one``, but the existing
    true_type is never shown, and the result goes to ``relabel_type`` via
    ``FailureEvent.relabel`` rather than ``label``. Returns 'labelled',
    'skipped', or 'quit'."""
    _print_event(event)
    print("[RELABEL MODE -- blind second pass, prior label is hidden]")
    _print_menu()

    choice = _prompt_type()
    if choice == "q":
        return "quit"
    if choice == "s":
        return "skipped"

    event.relabel(TYPE_KEYS[choice], relabelled_by=labelled_by, relabelled_at=_now_iso())
    store.write(event)
    return "labelled"


def run(store_root: Path, labelled_by: str | None = None) -> None:
    store = EventStore(store_root)
    labelled_by = labelled_by or getpass.getuser()

    progress = store.labelling_progress()
    print(f"Store: {store_root}")
    print(f"Total events: {progress['total']}  labelled: {progress['labelled']}  "
          f"unlabelled: {progress['unlabelled']}")
    if progress["by_type"]:
        print("By type so far:", progress["by_type"])

    pending = list(store.unlabelled())
    if not pending:
        print("Nothing to label.")
        return

    n_labelled = 0
    for event in pending:
        outcome = label_one(store, event, labelled_by)
        if outcome == "quit":
            break
        if outcome == "labelled":
            n_labelled += 1

    store.rebuild_index()
    print(f"\nSession done. Labelled {n_labelled} event(s) this run.")
    progress = store.labelling_progress()
    print("Updated totals:", progress)
    if progress["unsure_fraction"] is not None:
        print(f"Unsure fraction (upper bound on achievable classifier accuracy signal): "
              f"{progress['unsure_fraction']:.1%}")


def run_relabel(
    store_root: Path, n: int = 50, seed: Optional[int] = None, labelled_by: str | None = None
) -> None:
    """Sample up to ``n`` already-labelled events (deterministic given
    ``seed``) and re-present them blind for a second pass, then report
    Cohen's kappa between the two labels."""
    store = EventStore(store_root)
    labelled_by = labelled_by or getpass.getuser()

    pool = list(store.labelled())
    if not pool:
        print("No labelled events to sample from -- run normal labelling first.")
        return

    rng = random.Random(seed)
    sample = rng.sample(pool, min(n, len(pool)))
    print(f"Store: {store_root}")
    print(f"Relabelling {len(sample)} of {len(pool)} labelled event(s), seed={seed}.")

    n_relabelled = 0
    for event in sample:
        outcome = relabel_one(store, event, labelled_by)
        if outcome == "quit":
            break
        if outcome == "labelled":
            n_relabelled += 1

    store.rebuild_index()
    print(f"\nRelabel session done. Relabelled {n_relabelled} event(s) this run.")
    result = compute_relabel_kappa(store)
    if result["kappa"] is None:
        print(f"Cohen's kappa: not enough relabelled events yet (n={result['n']}, need >=2).")
    else:
        print(f"Cohen's kappa: {result['kappa']:.3f}  "
              f"(raw agreement {result['agreement_rate']:.1%}, n={result['n']})")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live-pilot event labelling CLI")
    parser.add_argument("store_root", type=Path)
    parser.add_argument("--relabel", action="store_true", help="run blind relabel/agreement mode")
    parser.add_argument("--n", type=int, default=50, help="relabel mode: sample size (default 50)")
    parser.add_argument("--seed", type=int, default=None, help="relabel mode: sampling seed")
    parser.add_argument("--labelled-by", type=str, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    if args.relabel:
        run_relabel(args.store_root, n=args.n, seed=args.seed, labelled_by=args.labelled_by)
    else:
        run(args.store_root, labelled_by=args.labelled_by)
