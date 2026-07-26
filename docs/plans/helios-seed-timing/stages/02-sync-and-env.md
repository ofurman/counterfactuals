# Stage 2: Sync code and build the cluster environment

**Goal**: Get the Stage 1 code onto Helios, confirm the live grant/storage picture still
matches `slurm/cluster.env`, and build the aarch64 Python 3.11 environment on a compute node.
**Dependencies**: Stage 1 DONE and committed — sync must carry the new timing column.

---

## Credentials

`.env` (gitignored, mode 600) holds `PLG_LOGIN`, `PLG_PASSWORD`, `PLG_GRANT`. Per
`plgrid-run/SKILL.md` rule 1, **only `PLG_LOGIN` may be used**, and nothing from `.env` may
be printed, committed, or copied to PLGrid. Authenticate with an SSH key/agent:

```bash
set -a; . ./.env; set +a      # exports PLG_LOGIN; do NOT echo PLG_PASSWORD
ssh -o BatchMode=yes "$PLG_LOGIN@login01.helios.cyfronet.pl" true
```

`BatchMode=yes` fails fast instead of prompting. If it fails, that is an **external
blocker**: add a Backlog entry ("SSH key not authorized on Helios — user must run
`ssh-copy-id` or start an agent") and mark this stage BLOCKED. Do **not** attempt password
automation, `sshpass`, or an interactive prompt.

---

## Steps

1. Push code, splits and dataset configs.
   - Command: `PLG_LOGIN="$PLG_LOGIN" ./slurm/sync-code.sh`
   - Details: The script creates `~/projects/counterfactuals` remotely and rsyncs with
     `.env`, `.git`, `.venv`, `/models`, `/outputs`, `/results` excluded. There is
     deliberately **no `--delete`** (macOS openrsync lacks `--delete-excluded`, and omitting
     deletion means a re-sync can never remove results already on the cluster). Re-running
     this stage is therefore safe.
   - The heavy excludes are anchored (`/models`, not `models`) because an unanchored pattern
     also matches the `counterfactuals/models/` package and ships a broken install
     (`ModuleNotFoundError: No module named 'counterfactuals.models'`). If you edit the
     exclude list, keep the anchors.

2. Inspect the live cluster state **without clobbering the verified config**.
   - Command (on the login node): `./slurm/preflight.sh` — **report only, no `--write`**
   - Details: `slurm/cluster.env` already holds values verified live on 2026-07-26.
     `preflight.sh --write` would overwrite it, potentially with `CHANGE-ME` placeholders,
     and `submit-all.sh` refuses to run on those. Run the report and **diff by eye** against
     the committed `cluster.env`:
     - `hpc-grants` still lists `plgcountercontex` with a gpu-gh200 allocation
     - `hpc-fs` still lists `plggcfsgenwro`, and `$PLG_GROUPS_STORAGE` is defined
     - `sinfo` shows `plgrid-gpu-gh200` up, and its `MaxTime` allows the 24 h CCHVAE request
       (if it does not, that is a Stage 3 finding — record it and cap `WALLTIME[cchvae]` there)
   - Only if a value genuinely changed: edit `cluster.env` by hand, in one commit, with the
     old value and the reason in the commit body. Never regenerate it wholesale.

3. Create the group-storage layout and project symlinks.
   - Command: `./slurm/bootstrap-storage.sh`
   - Details: Creates `$PLG_GROUPS_STORAGE/plggcfsgenwro/$USER/counterfactuals/{envs,cache,results}`
     and links `project/.venv` and `project/.cache` into it. Per plgrid-run rules 4-5 it must
     refuse to overwrite an existing path or an unexpected symlink — if it refuses, inspect
     the existing target rather than forcing it; a stale link from a prior attempt is a
     Backlog item, not something to `rm -rf` blindly.

4. Build the environment on a compute node.
   - Command:
     ```bash
     sbatch --account=plgcountercontex-gpu-gh200 --partition=plgrid-gpu-gh200 \
            --qos=now slurm/setup-env.sbatch
     ```
   - Details: This **must** run on a compute node, not the login node: Helios login nodes are
     x86_64 while GH200 compute is aarch64, so login-built wheels are the wrong
     architecture. `--qos=now` starts immediately and one job is all this needs.
   - Python is pinned to 3.11 on purpose. ML-bundle/25.10 ships 3.13, and
     `alibi → spacy → blis<0.8.0` has no cp313 aarch64 wheel, so pip compiles blis 0.7.11
     from source and dies in Cython (`CompileError: blis/py.pyx`). The script self-heals a
     stale 3.13 env by detecting the version and rebuilding.
   - `uv` is not in the module set; the script bootstraps it into a throwaway venv and then
     runs `uv sync --frozen --python 3.11`.

5. Wait for the build and read its log.
   - Commands: `sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode` and
     `cat slurm/logs/setup-env-<jobid>.out`
   - Details: The script's own tail prints `machine:` (expect `aarch64`), python 3.11, numpy
     and torch versions, then `seeding import OK` and `environment ready`. Absence of
     `environment ready` means failure regardless of the exit code you think you saw.

---

## Verification

- [ ] `ssh -o BatchMode=yes "$PLG_LOGIN@login01.helios.cyfronet.pl" 'ls ~/projects/counterfactuals/slurm'`
      lists the sbatch files
- [ ] Stage 1's change actually arrived:
      `ssh … 'grep -c cf_model_train_time ~/projects/counterfactuals/counterfactuals/pipelines/run_cchvae_traintest_pipeline.py'`
      is non-zero
- [ ] `.env` was **not** transferred: `ssh … 'test ! -e ~/projects/counterfactuals/.env'` succeeds
- [ ] `sacct` for the setup job shows `COMPLETED` / `0:0`, and its log ends with
      `environment ready`
- [ ] `ssh … '~/projects/counterfactuals/.venv/bin/python -c "import platform,torch;
      print(platform.machine(), platform.python_version(), torch.__version__)"'`
      prints `aarch64 3.11.x …`
- [ ] The package imports on the cluster:
      `ssh … 'cd ~/projects/counterfactuals && .venv/bin/python -c "import counterfactuals.models; print(\"ok\")"'`
      — guards against the unanchored-exclude failure mode from step 1
- [ ] `git diff --stat slurm/cluster.env` is empty, unless step 2 found a genuine change

---

## Commit

Only if step 2 found a real drift, or step 4 needed a fix:

`chore(slurm): reconcile cluster.env with live Helios state`

Otherwise no commit — this stage changes the cluster, not the repo. Record the setup job ID
in the index tracker notes.
