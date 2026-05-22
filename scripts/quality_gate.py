"""Repository quality gate for NeuroDrift Lab."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def count_files(path: str, pattern: str) -> int:
    return sum(1 for _ in (ROOT / path).glob(pattern))


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def main() -> int:
    checks = {
        "typed_package": exists("src/neurodrift/session.py")
        and exists("src/neurodrift/__init__.py"),
        "riemannian_metrics": exists("src/neurodrift/metrics/riemannian.py"),
        "trajectory_metrics": exists("src/neurodrift/metrics/trajectory.py"),
        "eval_environment": exists("src/neurodrift/envs/intent_drift.py"),
        "real_eeg_fixture": exists("data/fixtures/milimbeeg_s1.npz"),
        "real_neural_fixture": exists("data/fixtures/nlb_mc_maze_small_20.npz"),
        "dataset_loaders": exists("src/neurodrift/datasets/milimbeeg.py")
        and exists("src/neurodrift/datasets/nlb.py"),
        "docs_cover_methods": exists("docs/metrics.md") and exists("docs/data-fixtures.md"),
        "tests_are_not_token": count_files("tests", "test_*.py") >= 8,
        "readme_states_thesis": all(
            term in read("README.md").lower()
            for term in ("intent drift", "riemannian", "real")
        ),
    }
    scanned = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for base in ("src", "tests", "docs", "examples")
        for path in (ROOT / base).rglob("*")
        if path.is_file() and path.suffix in {".py", ".md"}
    ).lower()
    checks["no_placeholder_language"] = not any(
        marker in scanned for marker in ("todo", "lorem ipsum", "placeholder implementation")
    )
    payload = {
        "repo": "neurodrift-lab",
        "passed": all(checks.values()),
        "checks": checks,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
