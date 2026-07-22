# Raw datasets

Raw high-speed camera frames live here, one directory per dataset. **The image
files themselves are deliberately not tracked in git** (see `.gitignore`) — they
are large binaries that would bloat the repository history irreversibly.

## Expected layout

```
data/raw/<dataset>/
├── 1.tif
├── 2.tif
└── ...          # numbered consecutively from 1, in time order
```

The directory name is what you pass as `dataset=`:

```bash
naviernet stage=all dataset=highest_t
```

## Adding a new dataset

1. Drop the numbered TIFFs into `data/raw/<name>/`.
2. Copy `configs/experiment/highest_t.yaml` to `configs/experiment/<name>.yaml`
   and set the operating conditions and frame counts for that run.
3. Run it:

   ```bash
   naviernet stage=all dataset=<name> experiment=<name>
   ```

If segmentation misbehaves on the new frames, the parameters to reach for are
`imaging.dark_thresh`, `imaging.open_kernel`, and `imaging.wall_search_rows`.
Check `data/processed/<name>/qc_preprocess.png` before trusting anything
downstream.

## Datasets in use

| Dataset | Frames | Conditions |
| --- | --- | --- |
| `highest_t` | 12 (1–10 usable as one growth event) | FC-72, 2 W/cm², 5 mL/hr, 0.5 ms between frames |
