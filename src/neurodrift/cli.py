"""Command line entry point for NeuroDrift experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from neurodrift.benchmarks import ALIGNER_REGISTRY, run_alignment_benchmark
from neurodrift.datasets.milimbeeg import fetch_milimbeeg_sample, load_milimbeeg_pair
from neurodrift.datasets.nlb import load_nlb_h5_pair
from neurodrift.envs.intent_drift import IntentDriftEnv
from neurodrift.io import load_session_pair_npz, save_json, save_session_pair_npz
from neurodrift.simulation import SimulationConfig, simulate_session_pair


def _load_config(path: Path | None) -> SimulationConfig:
    if path is None:
        return SimulationConfig()
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return SimulationConfig(**data)


def _load_pair(args: argparse.Namespace):
    if args.input is not None:
        return load_session_pair_npz(args.input)
    return simulate_session_pair(_load_config(args.config))


def _emit(payload: dict[str, Any], output: Path | None) -> None:
    if output is not None:
        save_json(payload, output)
    print(json.dumps(payload, indent=2))


def run_simulation(args: argparse.Namespace) -> int:
    aligner_cls = ALIGNER_REGISTRY[args.aligner]
    pair = _load_pair(args)
    result = IntentDriftEnv(pair).evaluate(aligner_cls())
    _emit(result.to_dict(), args.output)
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    pair = _load_pair(args)
    benchmark = run_alignment_benchmark(pair)
    _emit(benchmark.to_dict(), args.output)
    return 0


def run_fetch_milimbeeg(args: argparse.Namespace) -> int:
    fetch_milimbeeg_sample(args.dest, subject=args.subject)
    pair = load_milimbeeg_pair(args.dest, subject=args.subject)
    save_session_pair_npz(pair, args.output)
    print(json.dumps({"subject": args.subject, "output": str(args.output)}, indent=2))
    return 0


def run_convert_nlb(args: argparse.Namespace) -> int:
    pair = load_nlb_h5_pair(args.input, dataset=args.dataset, max_trials=args.max_trials)
    save_session_pair_npz(pair, args.output)
    print(
        json.dumps(
            {"dataset": args.dataset, "input": str(args.input), "output": str(args.output)},
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neurodrift")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="run a synthetic drift evaluation")
    simulate.add_argument("--config", type=Path, default=None)
    simulate.add_argument("--input", type=Path, default=None, help="load a session pair NPZ")
    simulate.add_argument("--output", type=Path, default=None, help="write JSON report")
    simulate.add_argument("--aligner", choices=sorted(ALIGNER_REGISTRY), default="whiten-color")
    simulate.set_defaults(func=run_simulation)

    benchmark = subparsers.add_parser("benchmark", help="compare all alignment strategies")
    benchmark.add_argument("--config", type=Path, default=None)
    benchmark.add_argument("--input", type=Path, default=None, help="load a session pair NPZ")
    benchmark.add_argument("--output", type=Path, default=None, help="write JSON report")
    benchmark.set_defaults(func=run_benchmark)

    milimbeeg = subparsers.add_parser(
        "fetch-milimbeeg",
        help="download a small public MILimbEEG subset and convert it to NPZ",
    )
    milimbeeg.add_argument("--dest", type=Path, default=Path("data/external/milimbeeg"))
    milimbeeg.add_argument("--subject", default="S1")
    milimbeeg.add_argument("--output", type=Path, default=Path("data/fixtures/milimbeeg_s1.npz"))
    milimbeeg.set_defaults(func=run_fetch_milimbeeg)

    nlb = subparsers.add_parser("convert-nlb", help="convert an NLB HDF5 group to NPZ")
    nlb.add_argument("--input", type=Path, required=True)
    nlb.add_argument("--dataset", default="mc_maze_small_20")
    nlb.add_argument("--max-trials", type=int, default=80)
    nlb.add_argument("--output", type=Path, default=Path("data/fixtures/nlb_pair.npz"))
    nlb.set_defaults(func=run_convert_nlb)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
