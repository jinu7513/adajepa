# PushT data sampling audit

## Verified from the loader code

The map-style `TrajSlicerDataset` enumerates every start index `0..T-4*5` for each physical episode (`num_hist=3`, `num_pred=1`, `frameskip=5`). Window starts have **raw-frame stride 1**, so consecutive windows overlap heavily. The four observed frames in a window are at `s,s+5,s+10,s+15`. The DataLoader uses `shuffle=False`, `drop_last=False` (default), while `TrajSlicerDataset` permutes its window list once with NumPy. With one GPU/process, each window appears **once in a completed epoch without replacement**; an episode contributes many windows. The same window is ordinarily visited three times over three completed epochs.

Track A/B runs now save a partial checkpoint every 2,000 completed batches. An interrupted epoch resumes at that checkpoint's next batch. The loader still reads and skips earlier batches, but they are not re-optimized; batches between the checkpoint and preemption are re-optimized. `sampling_progress.jsonl` records the Slurm job, epoch and approximate visited-window count every 100 batches; pair-count JSON files record each epoch's planned appearance draws. The random color schemes key their draw by seed, epoch, physical episode ID and window start, so their colors can change between epochs but remain reproducible across workers and restarts. A replay of the same epoch uses the same seeded color pair and is explicitly a repeat, not a fresh independent sample.

## Counts requiring the real CIRCE dataset

Source episode count, exact train-window count, episode-length distribution, unique physical-trajectory hashes, per-episode window counts, batch count, and epoch-specific pair histograms **remain pending** in this local checkout. The P/J Slurm log previously showed `61,929` train batches for one epoch at batch size 32, but this does not establish the exact total window count or all episode frequencies. Do not infer independent trajectories from minibatch count.

Run `scripts/audit_two_track_data.py` with the real `data_full/pusht_noise` root on CIRCE. It instantiates the repository's real loader, writes `data_sampling_audit.json` and a populated `data_sampling_audit.md`, and for Track A writes `appearance_pair_counts.json` for every requested epoch. The training process also writes an audit from its actual loader into each model run directory. No numerical result is invented here.
