# NeuroDrift Lab

Tools for measuring BCI intent drift across sessions.

The core question:

```text
when the neural manifold moves, does decoded intent still mean the same thing?
```

This repo gives you:

- weak intent supervision
- SPD/Riemannian covariance drift
- subspace and Procrustes drift
- latent trajectory drift
- plasticity / meaning-gap reporting
- decoder-output drift
- alignment baselines
- meaning-preservation experiments
- a small eval environment
- simulated sessions
- real-data fixtures from MILimbEEG/OpenBCI and NLB

No giant notebook. No fake benchmark wrapper. Just typed code, tests, fixtures,
and commands that run.

## Run

```powershell
pip install -e .[dev,integrations]
pytest
ruff check .
```

Synthetic benchmark:

```powershell
neurodrift benchmark --config configs\simulated_medium_drift.json
```

Real EEG fixture:

```powershell
neurodrift benchmark --input data\fixtures\milimbeeg_s1.npz
```

Real neural-population fixture:

```powershell
neurodrift benchmark --input data\fixtures\nlb_mc_maze_small_20.npz
```

Meaning-gap report:

```powershell
python examples\plasticity_signal_report.py
```

Thesis experiment:

```powershell
neurodrift thesis --config configs\simulated_alignment_meaning_split.json
```

Evidence suite:

```powershell
neurodrift evidence --suite synthetic --compact --figure-dir artifacts\evidence
```

## Shape

```text
src/neurodrift/
  metrics/    drift measures
  intent.py   probabilistic weak intent labels
  plasticity.py meaning-gap signal
  experiments/ paper-facing thesis checks
  evidence.py repeated suite orchestration
  stats.py    bootstrap intervals and effect summaries
  figures.py  SVG evidence figures
  models/     decoders and aligners
  envs/       eval loop
  datasets/   real-data loaders
  adapters/   optional external bridges
tests/        numerical and flow tests
examples/     runnable reports
docs/         details when needed
```

## Useful Commands

```powershell
neurodrift simulate --aligner procrustes
neurodrift benchmark --config configs\simulated_high_drift.json
neurodrift thesis --input data\fixtures\nlb_mc_maze_small_20.npz
neurodrift evidence --suite all --compact
neurodrift fetch-milimbeeg --subject S1
neurodrift convert-nlb --input path\to\eval_data_test.h5
```

## License

MIT.
