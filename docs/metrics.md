# Metrics

## Affine-invariant covariance distance

The core SPD distance is:

```text
|| log(A^(-1/2) B A^(-1/2)) ||_F
```

This is useful when covariance matrices represent session-level neural
structure and we want the measurement to respect the SPD manifold.

## Log-Euclidean covariance distance

This compares covariance matrices after mapping them through the SPD matrix log.
It is cheaper and useful as a baseline against affine-invariant geometry.

## Subspace distance

Dominant neural subspaces are compared through principal angles. This captures
whether population activity occupies the same low-dimensional region even when
individual channels vary.

## Procrustes error

Paired trial features are aligned with the best orthogonal transform. Residual
error is a direct estimate of how much cannot be explained by simple manifold
rotation.

