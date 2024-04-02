import logging
import os

from time import time
import hydra
import neptune
from neptune.utils import stringify_unsupported
from hydra.utils import instantiate
from omegaconf import DictConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(
    config_path="conf", config_name="config_train_gen_model", version_base="1.2"
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
    gen_model_name = cfg.gen_model.model._target_.split(".")[-1]
    output_folder = os.path.join(cfg.experiment.output_folder, dataset_name)
    if cfg.disc_model is not None:
        disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
        disc_model_path = os.path.join(
            output_folder, f"disc_model_{disc_model_name}.pt"
        )
        gen_model_path = os.path.join(
            output_folder,
            f"gen_model_{gen_model_name}_relabeled_by_{disc_model_name}.pt",
        )
    else:
        gen_model_path = os.path.join(output_folder, f"gen_model_{gen_model_name}.pt")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs("results/model_train/", exist_ok=True)
    logger.info("Creatied output folder %s", output_folder)

    # Log parameters using Hydra config
    logger.info("Logging parameters")
    run["parameters/experiment"] = cfg.experiment
    run["parameters/dataset"] = dataset_name
    run["parameters/disc_model"] = stringify_unsupported(cfg.disc_model)
    run["parameters/gen_model"] = stringify_unsupported(cfg.gen_model)

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)

    if cfg.disc_model is not None:
        logger.info("Loading discriminator model %s", disc_model_name)
        disc_model = instantiate(
            cfg.disc_model.model, input_size=dataset.X_train.shape[1], target_size=1
        )
        disc_model.load(disc_model_path)
        dataset.y_train = disc_model.predict(dataset.X_train)
        dataset.y_test = disc_model.predict(dataset.X_test)
        train_dataloader = dataset.train_dataloader(
            batch_size=cfg.gen_model.batch_size,
            shuffle=True,
            noise_lvl=cfg.gen_model.noise_lvl,
        )
        test_dataloader = dataset.test_dataloader(
            batch_size=cfg.gen_model.batch_size, shuffle=False
        )

    logger.info("Training generative model")
    time_start = time()
    train_dataloader = dataset.train_dataloader(
        batch_size=cfg.gen_model.batch_size,
        shuffle=True,
        noise_lvl=cfg.gen_model.noise_lvl,
    )
    test_dataloader = dataset.test_dataloader(
        batch_size=cfg.gen_model.batch_size, shuffle=False
    )
    gen_model = instantiate(
        cfg.gen_model.model, features=dataset.X_train.shape[1], context_features=1
    )
    gen_model.fit(
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        num_epochs=cfg.gen_model.epochs,
        patience=cfg.gen_model.patience,
        learning_rate=cfg.gen_model.lr,
        checkpoint_path=gen_model_path,
        neptune_run=run,
    )
    run["metrics/eval_time"] = time() - time_start
    gen_model.save(gen_model_path)
    run["gen_model"].upload(gen_model_path)

    logger.info("Evaluating generative model")
    train_ll = gen_model.predict_log_prob(train_dataloader).mean()
    test_ll = gen_model.predict_log_prob(test_dataloader).mean()
    run["metrics/test_ll"] = test_ll
    run["metrics/train_ll"] = train_ll
    # TODO: Evaluate generative model
    # report = gen_model.test_model(test_loader=test_dataloader)
    # print(report)
    # run["metrics"] = process_classification_report(report, prefix="gen_test_orig")
    run.stop()


if __name__ == "__main__":
    main()
