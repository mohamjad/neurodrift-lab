# Data Fixtures

The repo includes small real-data fixtures for smoke testing. They are not
intended to replace full dataset downloads.

## `data/fixtures/milimbeeg_s1.npz`

Converted from a small subject subset of the public MILimbEEG/OpenBCI CSV
recordings.

- Source session: subject S1 motor imagery trials.
- Target session: subject S1 motor execution trials.
- Shape: EEG trials with 16 channels and 125 Hz sampling metadata.
- Purpose: real multi-session EEG drift smoke test.

Refresh:

```powershell
neurodrift fetch-milimbeeg --dest data\external\milimbeeg --subject S1 --output data\fixtures\milimbeeg_s1.npz
```
