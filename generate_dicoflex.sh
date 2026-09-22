#!/bin/bash
# Generate DiCoFlex counterfactuals from the WCSS-trained checkpoints.
set -e
cd ~/genwro/counterfactuals
export PATH=$HOME/.local/bin:$PATH
for DS in adult bank default gmc lending-club; do
  case $DS in lending-club) STEM=lending_club_split;; *) STEM=${DS}_split;; esac
  for SEED in 42 43 44; do
    echo "=== DiCoFlex $DS seed$SEED ==="
    uv run python -m counterfactuals.pipelines.run_dicoflex_traintest_pipeline \
      --config-name=dictum_dicoflex_config gen_model=orig_maf \
      experiment.seed=$SEED experiment.output_folder=results/wcss/seed_$SEED \
      disc_model.train_model=false gen_model.train_model=false \
      dataset.config_path=config/datasets/${STEM}.yaml \
      dataset.train_data_path=data_train_test_val/${DS}/train.csv \
      dataset.test_data_path=data_train_test_val/${DS}/test.csv \
      dataset.val_data_path=data_train_test_val/${DS}/val.csv \
      hydra.run.dir=results/wcss/hydra/${STEM}_${SEED}
  done
done
echo "=== score DiCoFlex ==="
uv run python -m scripts.compute_dictum_metrics --results-root results/wcss \
  --generation-scaler minmax_qt --scaler standard --metric-encoding ordinal \
  --datasets adult bank default gmc lending-club --seeds 42 43 44 \
  --methods DiCoFlex --output results/wcss/dicoflex_final
