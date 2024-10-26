import os
import logging
from omegaconf import DictConfig
import neptune
from neptune.utils import stringify_unsupported


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def log_parameters(cfg: DictConfig, run: neptune.Run):
    # Log parameters using Hydra config
    logger.info("Logging parameters")
    run["parameters/experiment"] = cfg.experiment
    run["parameters/dataset"] = cfg.dataset._target_.split(".")[-1]

    if cfg.get("disc_model"):
        run["parameters/disc_model/model_name"] = cfg.disc_model.model._target_.split(
            "."
        )[-1]
        run["parameters/disc_model"] = stringify_unsupported(cfg.disc_model)

    if cfg.get("gen_model"):
        run["parameters/gen_model/model_name"] = cfg.gen_model.model._target_.split(
            "."
        )[-1]
        run["parameters/gen_model"] = stringify_unsupported(cfg.gen_model)

    run["parameters/counterfactuals"] = cfg.counterfactuals_params
    run["parameters/counterfactuals/method_name"] = (
        cfg.counterfactuals_params.cf_method._target_.split(".")[-1]
    )
    run.wait()


def set_model_paths(cfg: DictConfig, fold: str = None):
    """
    Saves results in the output folder with the following structure:
    output_folder/dataset_name/_disc_model_name.pt
    output_folder/dataset_name/_gen_model_name.pt
    output_folder/dataset_name/method_name/results
    """
    # Set paths for saving models
    logger.info("Setting model paths")
    dataset_name = cfg.dataset._target_.split(".")[-1]
    gen_model_name = cfg.gen_model.model._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
    cf_method_name = cfg.counterfactuals_params.cf_method._target_.split(".")[-1]

    output_folder = os.path.join(
        os.path.abspath(cfg.experiment.output_folder), dataset_name
    )
    save_folder = os.path.join(output_folder, cf_method_name)
    if fold is not None:
        save_folder = os.path.join(save_folder, f"fold_{fold}")
        output_folder = os.path.join(output_folder, f"fold_{fold}")
    os.makedirs(output_folder, exist_ok=True)
    logger.info("Creatied output folder %s", output_folder)
    os.makedirs(save_folder, exist_ok=True)
    logger.info("Created save folder %s", save_folder)

    disc_model_path = os.path.join(output_folder, f"disc_model_{disc_model_name}.pt")
    if cfg.experiment.relabel_with_disc_model:
        gen_model_path = os.path.join(
            output_folder,
            f"gen_model_{gen_model_name}_relabeled_by_{disc_model_name}.pt",
        )
    else:
        gen_model_path = os.path.join(output_folder, f"gen_model_{gen_model_name}.pt")

    logger.info("Disc model path: %s", disc_model_path)
    logger.info("Gen model path: %s", gen_model_path)

    return disc_model_path, gen_model_path, save_folder
