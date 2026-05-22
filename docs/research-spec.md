# Research Spec

NeuroDrift tests one claim:

```text
cross-session alignment is not enough if decoded intent stops meaning the same thing
```

## Object

A session pair:

```text
source neural, source intent
target neural, target intent
```

The source session fits the decoder. The target session tests whether the
decoder still preserves intent after neural geometry moves.

## Primary Experiment

```powershell
neurodrift thesis --config configs\simulated_alignment_meaning_split.json
```

The experiment scores every registered alignment method on two axes:

- decoder error against target intent
- weak-intent semantic distance against the target session

The split matters. If an aligner improves MSE while increasing weak-intent
distance, it over-aligned. That is the failure mode this repo exists to expose.

The `simulated_alignment_meaning_split` preset is intentionally small and
deterministic. It is not a claim about biology. It is a fixture for the harder
measurement question: can the evaluator notice when an apparently better
alignment is worse for weak target meaning?

## Metrics

- `alignment_gain`: target decoder MSE improvement after alignment.
- `aligned_meaning_distance`: Jensen-Shannon distance between aligned decoded
  intent and target-session weak intent.
- `meaning_gain`: raw semantic distance minus aligned semantic distance.
- `meaning_gap`: plasticity signal combining geometry shift, trajectory shift,
  weak intent shift, and ambiguity.
- `hard_label_loss`: information lost by collapsing weak labels into hard labels.

## Falsification

The thesis weakens if:

- the best MSE method is always the best meaning-preservation method
- weak-label ambiguity does not predict cross-session failure
- geometry shift never separates from intent shift
- real data does not show any alignment/meaning split

The thesis strengthens if:

- MSE and meaning preservation disagree under plausible drift
- high ambiguity predicts relabeling pressure
- Riemannian movement appears before decoder failure
- real fixtures reproduce the same split at smaller scale

## Repo Boundary

This repo is the reference implementation. Adjacent repos should only split out
when the artifact has a different reader:

- `neurodrift-paper`: memo, figures, claims, limitations.
- `neurodrift-bench`: benchmark task definitions and scoring spec.
- `neurodrift-experiments`: heavier notebooks and external model runs.
