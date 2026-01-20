uv run python ./counterfactuals/pipelines/run_ppcefr_pipeline.py --multirun \
  dataset.config_path=config/datasets/toy_regression.yaml \
  disc_model=nn_regression,linear_regression &

uv run python ./counterfactuals/pipelines/run_ppcefr_pipeline.py --multirun \
  dataset.config_path=config/datasets/concrete.yaml \
  disc_model=nn_regression,linear_regression &

uv run python ./counterfactuals/pipelines/run_ppcefr_pipeline.py --multirun \
  dataset.config_path=config/datasets/diabetes.yaml \
  disc_model=nn_regression,linear_regression &

uv run python ./counterfactuals/pipelines/run_ppcefr_pipeline.py --multirun \
  dataset.config_path=config/datasets/scm20d.yaml \
  disc_model=nn_regression,linear_regression &

uv run python ./counterfactuals/pipelines/run_ppcefr_pipeline.py --multirun \
  dataset.config_path=config/datasets/yacht.yaml \
  disc_model=nn_regression,linear_regression &