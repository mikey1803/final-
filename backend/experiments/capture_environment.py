#!/usr/bin/env python
"""Task 14 Part 3: pin the exact environment every result in the paper
was produced under -- Python version, every installed dependency's
exact version, every LLM model string actually called (provider, model
ID, and any resolved version suffix the API disclosed), and the dates
those calls happened.

Reuses ``runtime_benchmark.py::capture_hardware_info`` rather than
re-deriving hardware/OS capture logic. Dependency versions come from
``pip freeze`` against the SAME interpreter used to run every experiment
in this repository (the project's ``.venv``) -- not a guessed or
hand-maintained list, since ``requirements.txt`` itself pins no
versions (confirmed: every line is a bare package name).

Usage
-----
    python experiments/capture_environment.py
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from experiments.runtime_benchmark import capture_hardware_info  # noqa: E402

OUTPUT_DIR = _BACKEND_DIR / "results" / "tables"
LLM_DATA = _BACKEND_DIR / "results" / "tables" / "llm_designers.parquet"


def capture_pip_freeze() -> list[str]:
    """The exact installed package versions in the interpreter running
    this script -- the same interpreter every experiment in this
    session was run with. Real ``pip freeze`` output, not a
    hand-maintained/reconstructed list."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    )
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def capture_llm_model_pins() -> pd.DataFrame:
    """Provider, configured model ID, and the API-disclosed resolved
    ``model_version`` (which may just echo an alias back verbatim, e.g.
    Google's -- reported honestly either way, never assumed pinned when
    it wasn't), plus the real call-date range per family."""
    if not LLM_DATA.exists():
        return pd.DataFrame()
    df = pd.read_parquet(LLM_DATA)
    rows = []
    for family, sub in df.groupby("family"):
        ts = pd.to_datetime(sub["timestamp_utc"])
        rows.append({
            "family": family,
            "provider_type": sub["provider_type"].iloc[0],
            "model_id_configured": sub["model"].iloc[0],
            "model_version_api_disclosed": sorted(sub["model_version"].dropna().unique().tolist()),
            "n_calls": len(sub),
            "first_call_utc": ts.min().isoformat(),
            "last_call_utc": ts.max().isoformat(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    hw = capture_hardware_info()
    print("=== Hardware / OS ===")
    for k, v in hw.items():
        print(f"  {k}: {v}")

    packages = capture_pip_freeze()
    print(f"\n=== Installed packages ({len(packages)}), exact versions ===")
    for line in packages:
        print(f"  {line}")

    llm_pins = capture_llm_model_pins()
    print("\n=== LLM model strings actually called ===")
    if llm_pins.empty:
        print("  No LLM designer data found.")
    else:
        print(llm_pins.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hw_df = pd.DataFrame([hw])
    hw_df.to_parquet(OUTPUT_DIR / "environment_hardware.parquet", index=False)

    pkg_df = pd.DataFrame([line.split("==", 1) if "==" in line else [line, None] for line in packages],
                          columns=["package", "version"])
    pkg_df.to_parquet(OUTPUT_DIR / "environment_packages.parquet", index=False)

    if not llm_pins.empty:
        llm_pins.to_parquet(OUTPUT_DIR / "environment_llm_model_pins.parquet", index=False)

    # Also a single human-readable snapshot file, dated -- the canonical
    # "what was this run with" record referenced from the paper.
    snapshot_path = OUTPUT_DIR.parent / "environment_snapshot.txt"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        f.write("Environment snapshot -- captured by experiments/capture_environment.py\n")
        f.write(f"Capture date (UTC): {pd.Timestamp.utcnow().isoformat()}\n\n")
        f.write("=== Hardware / OS ===\n")
        for k, v in hw.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\n=== Installed packages ({len(packages)}) ===\n")
        for line in packages:
            f.write(f"{line}\n")
        f.write("\n=== LLM model strings actually called ===\n")
        if not llm_pins.empty:
            f.write(llm_pins.to_string(index=False))
            f.write("\n")
        else:
            f.write("No LLM designer data found.\n")
    print(f"\nWrote {snapshot_path}")
    print(f"Wrote {OUTPUT_DIR / 'environment_hardware.parquet'}")
    print(f"Wrote {OUTPUT_DIR / 'environment_packages.parquet'}")
    if not llm_pins.empty:
        print(f"Wrote {OUTPUT_DIR / 'environment_llm_model_pins.parquet'}")


if __name__ == "__main__":
    main()
