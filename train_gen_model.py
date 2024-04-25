import logging
import os
import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    logger.info("Initializing Neptune run")
    # run = neptune.init_run(
    #     mode="async" if cfg.neptune.enable else "offline",
    #     project=cfg.neptune.project,
    #     api_token=cfg.neptune.api_token,
    #     tags=list(cfg.neptune.tags) if "tags" in cfg.neptune else None,
    # )

    dataset_name = cfg.dataset._target_.split(".")[-1]
    gen_model_name = cfg.gen_model.model._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
    output_folder = os.path.join(cfg.experiment.output_folder, dataset_name)
    os.makedirs(output_folder, exist_ok=True)
    save_folder = os.path.join(output_folder, "ppcef")
    os.makedirs(save_folder, exist_ok=True)
    logger.info("Creatied output folder %s", output_folder)

    # Log parameters using Hydra config
    logger.info("Logging parameters")
    # run["parameters/experiment"] = cfg.experiment
    # run["parameters/dataset"] = cfg.dataset._target_.split(".")[-1]
    # run["parameters/disc_model/model_name"] = disc_model_name
    # run["parameters/disc_model"] = cfg.disc_model
    # run["parameters/gen_model/model_name"] = gen_model_name
    # run["parameters/gen_model"] = cfg.gen_model
    # run["parameters/counterfactuals"] = cfg.counterfactuals
    # run["parameters/experiment"] = cfg.experiment
    # run["parameters/dataset"] = dataset_name
    # run["parameters/disc_model"] = stringify_unsupported(cfg.disc_model)
    # run.wait()
    train_mean_nll = []
    test_mean_nll = []

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)
    for fold_n, (_, _, _, _) in enumerate(dataset.get_cv_splits(n_splits=5)):
        # disc_model_path = os.path.join(
        #     output_folder, f"disc_model_{fold_n}_{disc_model_name}.pt"
        # )
        if cfg.experiment.relabel_with_disc_model:
            gen_model_path = os.path.join(
                output_folder,
                f"gen_model_{fold_n}_{gen_model_name}_relabeled_by_{disc_model_name}.pt",
            )
        else:
            gen_model_path = os.path.join(
                output_folder, f"gen_model_{fold_n}_{gen_model_name}.pt"
            )

        logger.info("Training discriminator model")
        # num_classes = 1 if disc_model_name == "LogisticRegression" else len(np.unique(dataset.y_train))

        train_dataloader = dataset.train_dataloader(
            batch_size=cfg.disc_model.batch_size, shuffle=True, noise_lvl=0
        )
        test_dataloader = dataset.test_dataloader(
            batch_size=cfg.disc_model.batch_size, shuffle=False
        )
        # disc_model = instantiate(
        #     cfg.disc_model.model,
        #     input_size=dataset.X_train.shape[1],
        #     target_size=num_classes,
        # )
        # disc_model.load(disc_model_path)
        # disc_model.fit(
        #     train_dataloader,
        #     test_dataloader,
        #     epochs=cfg.disc_model.epochs,
        #     lr=cfg.disc_model.lr,
        #     patience=cfg.disc_model.patience,
        #     checkpoint_path=disc_model_path,
        # )
        # disc_model.save(disc_model_path)
        # logger.info("Evaluating discriminator model")
        # print(classification_report(dataset.y_test, disc_model.predict(dataset.X_test)))
        # report = classification_report(
        #     dataset.y_test, disc_model.predict(dataset.X_test), output_dict=True
        # )
        # pd.DataFrame(report).transpose().to_csv(
        #     os.path.join(save_folder, f"eval_disc_model_{fold_n}_{disc_model_name}.csv")
        # )
        # run[f"{fold_n}/metrics"] = process_classification_report(
        #     report, prefix="disc_test"
        # )

        # run[f"{fold_n}/disc_model"].upload(disc_model_path)

        # if cfg.experiment.relabel_with_disc_model:
        #     dataset.y_train = disc_model.predict(dataset.X_train)
        #     dataset.y_test = disc_model.predict(dataset.X_test)

        logger.info("Training generative model")
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
        # gen_model.load(gen_model_path)
        gen_model.fit(
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            num_epochs=cfg.gen_model.epochs,
            patience=cfg.gen_model.patience,
            learning_rate=cfg.gen_model.lr,
            checkpoint_path=gen_model_path,
            eps=1e-2,
            # neptune_run=run,
        )
        # run[f"{fold_n}/metrics/gen_model_train_time"] = time() - time_start
        gen_model.save(gen_model_path)
        # run[f"{fold_n}/gen_model"].upload(gen_model_path)

        train_ll = gen_model.predict_log_prob(train_dataloader).mean().item()
        test_ll = gen_model.predict_log_prob(test_dataloader).mean().item()
        pd.DataFrame({"train_ll": [train_ll], "test_ll": [test_ll]}).to_csv(
            os.path.join(save_folder, f"eval_gen_model_{fold_n}_{gen_model_name}.csv")
        )

        train_mean_nll.append(train_ll)
        test_mean_nll.append(test_ll)

    print("Train LL: ", np.mean(train_mean_nll), "+-", np.std(train_mean_nll))
    print("Test LL: ", np.mean(test_mean_nll), "+-", np.std(test_mean_nll))
    # run["metrics"].upload(os.path.join(output_folder, "metrics.csv"))
    # run.stop()


if __name__ == "__main__":
    main()
