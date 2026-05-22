# NeuroDrift Memo

## Claim

BCI alignment should not be judged by decoder performance alone.

When neural activity drifts across sessions, an alignment method can improve
target-session decoder error while making decoded intent less faithful to the
target session. That is the measurement problem NeuroDrift isolates.

## Why This Matters

Most BCI systems treat drift as nuisance variation:

```text
session changed -> align neural features -> recover decoder performance
```

That is necessary, but incomplete. If the brain is adapting, the target being
decoded may also be moving. In that regime, alignment can become an eraser: it
can make today's neural activity look like yesterday's decoder input while
discarding evidence that today's intent distribution is different.

The core question is not only:

```text
did the decoder improve?
```

It is:

```text
did decoded intent still mean the same thing?
```

## Measurement Object

NeuroDrift evaluates a source-target session pair:

```text
source neural, source intent
target neural, target intent
```

The source session fits the decoder. The target session tests whether alignment
preserves intent under neural geometry movement.

The repo scores:

- decoder MSE
- Riemannian covariance drift
- subspace and Procrustes drift
- latent trajectory drift
- weak-intent ambiguity
- meaning preservation
- over-alignment

## Minimal Result

Run:

```powershell
neurodrift thesis --config configs\simulated_alignment_meaning_split.json
```

Default deterministic result:

```text
best_by_mse: procrustes
best_by_meaning: identity
has_alignment_meaning_split: true
```

In this fixture, Procrustes alignment sharply improves target decoder MSE, but
weak-intent meaning distance gets worse. The command writes:

```text
artifacts/thesis/meaning-split.svg
```

That plot is the thesis in one figure:

```text
decoder error down, meaning distance up
```

## Evidence Protocol

Run:

```powershell
neurodrift evidence --suite synthetic --compact --figure-dir artifacts\evidence
```

The suite repeats the thesis experiment across seeds and drift levels, then
reports bootstrap intervals for:

- alignment/meaning split rate
- over-alignment rate
- over-alignment penalty
- hard-label loss
- source-target intent shift
- best MSE alignment gain
- best meaning distance

Fixture runs are also supported:

```powershell
neurodrift evidence --suite fixtures --compact
```

The included fixtures are smoke tests from MILimbEEG/OpenBCI and NLB-style
neural data. They are not final evidence. They make the pipeline concrete.

## What Would Strengthen This

- invasive human BCI sessions with intent labels rich enough for weak intent
  distributions
- NLB-style population runs with stronger session structure
- comparisons against CEBRA, LFADS/NDT-style latents, and manifold stabilization
- longitudinal runs where geometry shift appears before decoder failure
- evidence that ambiguity predicts relabeling pressure

## What Would Falsify This

- decoder MSE and meaning preservation always select the same method
- over-alignment disappears on real neural sessions
- weak-label ambiguity does not predict failure
- geometry drift never separates from intent drift

That is the point. The repo does not assume the thesis is true. It makes the
failure mode measurable.
