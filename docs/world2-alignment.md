# World2 Alignment

The next layer is not better denoising.

It is measuring intent under continuous neural change.

## Thesis

Noise, drift, and intent ambiguity are not separate problems. They are symptoms
of a moving substrate.

The brain changes. Electrodes drift. Labels lag the neural event. In speech BCI,
the ground truth is not just produced audio; it is what the person intended to
say.

`neurodrift-lab` now treats that as a measurement problem:

- neural geometry shift
- latent trajectory shift
- weak intent shift
- ambiguity
- meaning gap

## Added Surface

- `IntentDistribution`: probabilistic labels for weak intent supervision.
- `soft_intent_from_observations`: converts noisy observations into soft labels.
- `intent_distribution_distance`: measures intent movement.
- `PlasticitySignal`: joins neural drift and intent drift.
- `build_plasticity_signal`: produces the meaning-gap report.

## Why It Matters

Cross-session alignment should not always erase drift. Some drift is plasticity.
The job is to separate bad measurement movement from real meaning movement.

That is the line this repo is moving toward.
