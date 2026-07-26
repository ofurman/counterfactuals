# Stage 3: Smoke calibration

**Goal**: Run one real array task end-to-end on Helios and reconcile the extrapolated
walltimes against measured elapsed time before committing 60 tasks to the queue.
**Dependencies**: Stage 2 DONE (environment built and importable on aarch64).

---

## Why this gate exists

`slurm/README.md` is explicit that `WALLTIME` in `submit-all.sh` is extrapolated from *local*
`cf_search_time` values (CCHVAE on adult: 11586 s of CF search alone) plus headroom for
retraining that **has never been measured on this hardware**. The generative flow trains for
up to 2000 epochs on CPU and could dominate. Submitting 60 tasks against unvalidated
walltimes risks losing a multi-day sweep to TIMEOUT.

---

## Steps

1. Dry-run the smoke shape first (plgrid-run rule 6: `--test-only` before a new job shape).
   - Command: `./slurm/submit-all.sh --smoke`
   - Details: Prints the `sbatch --test-only` verdict. Any "Invalid account or
     account/partition combination" here means step 2 of Stage 2 missed a drift — go back.

2. Submit the smoke task.
   - Command: `./slurm/submit-all.sh --smoke --submit`
   - Details: One task — DiCoFlex on `default`, seed 0 (`--array=6`, since
     `dataset_idx*n_seeds + seed_idx = 2*3+0` and `default` is `DATASETS[2]`). Cheapest
     method x smallest dataset (860 factuals), 4 h request, `--qos=now` so it starts
     immediately. Results go to the `smoke` tag, isolated from `seeds`.
   - The job ID is appended to `slurm/logs/submitted.tsv`. Record it in the index tracker.
   - **If Stage 1 deferred its CCHVAE assertion here**, run a second smoke task for cchvae
     instead of relying on the dicoflex one — DiCoFlex writes `cf_model_train_time = 0.0` by
     design and therefore proves nothing about the VAE timer. Override with
     `--export="ALL,CF_METHOD=cchvae,CF_SEEDS=0 1 2,CF_RESULTS_TAG=smoke"` and `--array=6`.

3. Wait for it, then read the real numbers.
   - Commands:
     ```bash
     sacct -j <jobid> --format=JobID,JobName%20,State,Elapsed,MaxRSS,ExitCode,AllocTRES%40
     ```
   - Details: Do not poll tighter than every 5 minutes. Capture `Elapsed` (walltime
     calibration), `MaxRSS` (the sbatch requests 64 G — confirm that is not tight), and
     `AllocTRES`. The `AllocTRES` / `CPUTimeRAW` reading also answers the open question in
     `slurm/README.md` about whether a no-`--gres` job on the GPU partition is still billed
     as GPU-hours; record the answer in the index Decisions either way.

4. Break the elapsed time into its parts.
   - Details: Read
     `$PLG_GROUPS_STORAGE/plggcfsgenwro/$USER/counterfactuals/results/smoke/seed_0/default_split/DiCoFlex/fold_0/cf_metrics_SimpleMLPClassifier.csv`
     and note `disc_train_time`, `gen_train_time`, `cf_model_train_time`, `cf_search_time`.
     The point is to learn which half dominates on GH200 cores: if training dominates, the
     per-method `WALLTIME` spread in `submit-all.sh` (which is driven entirely by CF search
     cost) is mis-shaped and every method needs a training floor added.

5. Reconcile the walltimes.
   - File: `slurm/submit-all.sh`
   - Details: The `WALLTIME` map is `dice=08:00:00`, `cchvae=24:00:00`,
     `dicoflex=12:00:00`. Scale from the measurement: DiCoFlex on `default` is the cheapest
     cell, so extrapolate to the largest dataset (`adult`/`gmc`/`lending-club` have ~24k
     train rows vs `default`'s ~21.6k, and 10k test rows vs 3k — so CF search scales roughly
     3x on test size alone) and add the measured training floor. Keep ~50% headroom.
   - Confirm the partition `MaxTime` from `sinfo` permits the CCHVAE request. If `MaxTime` is
     below 24 h, cap `WALLTIME[cchvae]` at `MaxTime` and add a Backlog entry noting CCHVAE
     may need checkpointing or a dataset split across tasks.
   - If the smoke elapsed time indicates any method cannot finish inside the partition cap,
     do **not** silently submit anyway. Record it in the Backlog and proceed with the methods
     that fit.

---

## Verification

- [ ] `sacct` for the smoke job shows `State=COMPLETED` and `ExitCode=0:0`
- [ ] `cf_metrics_SimpleMLPClassifier.csv` exists under the `smoke` tag, is non-empty, and
      has a `validity` value > 0 (a completed job that produced degenerate results is not a
      calibration)
- [ ] `cf_model_train_time` is present in that CSV — proof the Stage 1 change is live on the
      cluster, not just in the repo
- [ ] `results/smoke/seed_0/default_split_dicoflex_run.json` exists and its `git_commit`
      matches the Stage 1 commit SHA — proof the sweep will run the instrumented code
- [ ] Every `WALLTIME` entry in `submit-all.sh` is either unchanged with a written
      justification in the index notes, or updated with the measurement it came from
- [ ] `./slurm/submit-all.sh` (full dry run, no `--submit`) reports all arrays accepted

---

## Commit

`chore(slurm): calibrate walltimes from Helios smoke run`

Include the measured elapsed/`MaxRSS` numbers in the commit body. If nothing needed
changing, skip the commit and record the measurements in the index tracker notes instead.
