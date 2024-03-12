import logging
import os
import hydra
import neptune
from neptune.utils import stringify_unsupported
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.metrics import classification_report
from counterfactuals.utils import process_classification_report

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(
    config_path="conf", config_name="config_train_disc_model", version_base="1.2"
)
def main(cfg: DictConfig):
    logger.info("Initializing Neptune run")
    run = neptune.init_run(
        mode="async" if cfg.neptune.enable else "offline",
        project=cfg.neptune.project,
        api_token=cfg.neptune.api_token,
        tags=list(cfg.neptune.tags) if "tags" in cfg.neptune else None,
    )

    dataset_name = cfg.dataset._target_.split(".")[-1]
    model_name = cfg.disc_model.model._target_.split(".")[-1]
    output_folder = os.path.join(cfg.experiment.output_folder, dataset_name)
    os.makedirs(output_folder, exist_ok=True)
    logger.info("Creatied output folder %s", output_folder)

    # Log parameters using Hydra config
    logger.info("Logging parameters")
    run["parameters/experiment"] = cfg.experiment
    run["parameters/dataset"] = dataset_name
    run["parameters/disc_model"] = stringify_unsupported(cfg.disc_model)

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)
    X_train = dataset.X_train

    logger.info("Training discriminator model")
    disc_model = instantiate(
        cfg.disc_model.model, input_size=X_train.shape[1], target_size=1
    )
    train_dataloader = dataset.train_dataloader(
        batch_size=cfg.disc_model.batch_size, shuffle=True, noise_lvl=0
    )
    disc_model.fit(train_dataloader, epochs=cfg.disc_model.epochs, lr=cfg.disc_model.lr)

    logger.info("Evaluating discriminator model")
    print(classification_report(dataset.y_test, disc_model.predict(dataset.X_test)))
    report = classification_report(
        dataset.y_test, disc_model.predict(dataset.X_test), output_dict=True
    )
    run["metrics"] = process_classification_report(report, prefix="disc_test")

    disc_model_path = os.path.join(output_folder, f"disc_model_{model_name}.pt")
    disc_model.save(disc_model_path)
    run["disc_model"].upload(disc_model_path)
    run.stop()


if __name__ == "__main__":
    main()
