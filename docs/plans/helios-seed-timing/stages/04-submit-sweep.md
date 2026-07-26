# Stage 4: Submit and monitor the sweep

**Goal**: Submit exactly 60 array tasks (3 methods x 5 datasets x 3 seeds, plus the DiCoFlex
`target_class=0` re-run) and verify every one produced a usable `cf_metrics` CSV.
**Dependencies**: Stage 3 DONE (walltimes calibrated, smoke `COMPLETED`).

---

## Scope: exactly 60 tasks, no constraints array

Per index Decision 2 the `cf-constraints` array is out of scope. `submit-all.sh` submits it
only when `--only` is empty, so use three `--only` invocations:

```bash
./slurm/submit-all.sh --submit --only dice        # cf-dice,         15 tasks
./slurm/submit-all.sh --submit --only cchvae      # cf-cchvae,       15 tasks
./slurm/submit-all.sh --submit --only dicoflex    # cf-dicoflex, 15 + cf-dicoflex-tc0, 15
```

The third also submits `cf-dicoflex-tc0`, because that block fires when
`ONLY` is empty **or** equals `dicoflex`. Total: 60. Each array is capped at
`MAX_CONCURRENT=8` concurrent tasks.

**Do not run bare `./slurm/submit-all.sh --submit`** — it would add the 5 constraint tasks.

## Idempotence guard — read before submitting

A double submission burns the grant twice and interleaves two writers into the same result
directories. Before each command:

```bash
squeue -u "$USER" -o '%i %j %t %M'      # nothing named cf-* should be pending/running
cat slurm/logs/submitted.tsv            # arrays already submitted in a prior run
```

If an array is already in `submitted.tsv` and appears in `squeue` or `sacct`, **skip its
submission** and go straight to monitoring. Resuming this stage after an interrupted run
means monitoring, not resubmitting.

---

## Steps

1. Dry-run each method shape.
   - Commands: `./slurm/submit-all.sh --only dice`, `… --only cchvae`, `… --only dicoflex`
   - Details: All arrays must be accepted. Note the `--test-only` start-time estimate; the
     `normal` QOS previously estimated ~4 days to start, so a long estimate is expected, not
     an error.

2. Submit the three method groups.
   - Details: Run the three `--submit --only …` commands above. Each prints
     `name<TAB>jobid` and appends to `slurm/logs/submitted.tsv`. Copy all four
     (name, job ID) pairs into the index tracker notes for stage 4 — that table is what a
     later resumed run uses to find the sweep.
   - Mail alerts go to the address in `cluster.env` at `BEGIN,END,FAIL`, one message per
     array (~8 messages total). Do **not** add `ARRAY_TASKS` to the mail type.

3. Monitor without busy-polling.
   - Command:
     ```bash
     sacct -X --format=JobID,JobName%20,State,Elapsed,ExitCode --starttime today
     ```
   - Details: Poll at **no tighter than 30-minute intervals**. This sweep can take days:
     CCHVAE may request up to 24 h per task, 8 concurrent, 15 tasks — two full waves
     minimum, on top of a queue start that may itself be days out.
   - **While waiting, go do Stage 5.** It depends only on the CSV schema, not on the sweep
     finishing, and it is the actual missing piece of software. Returning here afterwards is
     the intended path.
   - If Stage 5 is DONE and the sweep is still running: write the outstanding job IDs and
     the current `sacct` summary into the tracker notes, leave Stage 4 as **IN_PROGRESS**,
     and end the run cleanly. A later invocation reads IN_PROGRESS and resumes at step 4.

4. Verify per-task outcomes, not just array states.
   - Details: A job ID is not evidence of success (plgrid-run rule 8). For each of the 60
     tasks confirm `State=COMPLETED` and `ExitCode=0:0`, then confirm the artefact exists and
     is non-empty. Count the CSVs on the cluster:
     ```bash
     RES="$PLG_GROUPS_STORAGE/plggcfsgenwro/$USER/counterfactuals/results"
     find "$RES/seeds" "$RES/seeds-tc0" -name 'cf_metrics_*.csv' -size +0 | wc -l   # expect 60
     ```
   - Expected layout per tag:
     `results/<tag>/seed_{0,1,2}/<dataset>_split/<Method>/fold_0/cf_metrics_SimpleMLPClassifier.csv`
     where `<Method>` is `DiceExplainerWrapper`, `CCHVAE` or `DiCoFlex`, and `<dataset>_split`
     is `adult_split`, `bank_split`, `default_split`, `gmc_split`, `lending_club_split`.

5. Triage failures — partially, not all-or-nothing.
   - Details: For each `FAILED`, `TIMEOUT` or `OUT_OF_MEMORY` task, read
     `slurm/logs/cf-<method>-<arrayjobid>_<taskid>.err`, classify, and act:
     - **TIMEOUT** → raise that method's `WALLTIME` from the observed `Elapsed` of its
       siblings and resubmit only the affected `--array=<indices>` (comma-separated). One
       retry, then Backlog.
     - **OUT_OF_MEMORY** → raise `--mem` above the 64 G in `run-baselines.sbatch` for that
       resubmission only; note it in Fixed Issues.
     - **Anything else** → one fix attempt via subagent with the `.err` tail, then Backlog.
   - Per the index partial-sweep rule, missing cells do **not** block Stage 6. Log each
     missing (method, dataset, seed) triple in the Backlog and continue.

---

## Verification

- [ ] `slurm/logs/submitted.tsv` contains exactly four array entries: `seeds-dice`,
      `seeds-cchvae`, `seeds-dicoflex`, `seeds-dicoflex-tc0` — and **no** `constraints-adult`
- [ ] `sacct` shows 60 array tasks; every non-`COMPLETED` task is logged in the Backlog with
      its state
- [ ] The `find … | wc -l` count above is 60, or equals 60 minus the Backlog-logged failures
- [ ] Every CSV found has a `cf_model_train_time` column (proves the instrumented code ran)
- [ ] `results/<tag>/seed_<N>/<stem>_<method>_run.json` exists per task and its `git_commit`
      is the Stage 1 commit for all of them — a differing SHA means part of the sweep ran
      pre-instrumentation code and those cells must be flagged in the report
- [ ] Spot-check one CSV per method for `validity > 0`; a `COMPLETED` task with zero validity
      is a silent failure and belongs in the Backlog

---

## Commit

`chore(slurm): record submitted sweep job ids`

Commit only the index tracker notes and any `WALLTIME`/`--mem` change a retry required.
Do not commit result CSVs — they live in group storage and are pulled in Stage 6.
