"""Regression test for a Task 13 finding: ``run_matrix_family_sweep``
(``src/evaluation/matrix_family.py``, the driver behind PAPER_PREP.md's
Phase 3A and part of Phase 5) used to derive its per-episode simulation
seed partly from Python's built-in ``hash()`` on the algorithm name --
``hash()`` is randomized per-process for strings (``PYTHONHASHSEED``)
unless explicitly fixed, so the SAME call with the SAME ``base_seed`` and
``n_seeds`` produced genuinely different simulated results across
separate process invocations. Confirmed directly: two fresh interpreter
processes computing ``hash('ucb1') % 997`` returned 399 and 286.

Fixed by replacing ``hash(algo) % 997`` with ``zlib.crc32(algo.encode())
% 997`` (same purpose -- a stable per-algorithm-name seed offset -- but
deterministic across processes). This test runs the sweep as two
independent subprocesses (full interpreter isolation, so each gets its
own fresh, randomized ``PYTHONHASHSEED`` if the bug ever regressed) and
asserts byte-identical output.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent.parent

_RUN_SNIPPET = """
import sys
sys.path.insert(0, r"{backend_dir}")
from src.evaluation.matrix_family import run_matrix_family_sweep
df = run_matrix_family_sweep(gaps=[0.2], n_matrices=3, n_seeds=2, num_episodes=20)
df.to_csv(r"{out_path}", index=False)
"""


def _run_sweep_in_fresh_process(out_path: Path) -> None:
    snippet = _RUN_SNIPPET.format(backend_dir=str(_BACKEND_DIR), out_path=str(out_path))
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stdout}\n{result.stderr}"


def test_run_matrix_family_sweep_is_deterministic_across_processes(tmp_path):
    out1 = tmp_path / "run1.csv"
    out2 = tmp_path / "run2.csv"
    _run_sweep_in_fresh_process(out1)
    _run_sweep_in_fresh_process(out2)

    df1 = pd.read_csv(out1)
    df2 = pd.read_csv(out2)
    pd.testing.assert_frame_equal(df1, df2)
