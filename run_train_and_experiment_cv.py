import logging
import os
import hydra
import numpy as np
import pandas as pd
from time import time
import torch
import neptune
from neptune.utils import stringify_unsupported
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.metrics import classification_report
import torch.utils

from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.cf_methods.regional_ppcef import RPPCEF

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
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
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
    output_folder = os.path.join(cfg.experiment.output_folder, dataset_name)
    os.makedirs(output_folder, exist_ok=True)
    save_folder = os.path.join(output_folder, "ppcef")
    os.makedirs(save_folder, exist_ok=True)
    logger.info("Creatied output folder %s", output_folder)

    # Log parameters using Hydra config
    logger.info("Logging parameters")
    run["parameters/experiment"] = cfg.experiment
    run["parameters/disc_model/model_name"] = disc_model_name
    run["parameters/disc_model"] = cfg.disc_model
    run["parameters/gen_model/model_name"] = gen_model_name
    run["parameters/gen_model"] = cfg.gen_model
    run["parameters/counterfactuals"] = cfg.counterfactuals
    run["parameters/experiment"] = cfg.experiment
    run["parameters/dataset"] = dataset_name
    run["parameters/disc_model"] = stringify_unsupported(cfg.disc_model)
    run.wait()

    log_df = pd.DataFrame()

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)
    for fold_n, (_, _, _, _) in enumerate(dataset.get_cv_splits(n_splits=5)):
        disc_model_path = os.path.join(
            output_folder, f"disc_model_{fold_n}_{disc_model_name}.pt"
        )
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
        binary_datasets = [
            "MoonsDataset",
            "LawDataset",
            "HelocDataset",
            "AuditDataset",
        ]
        num_classes = (
            1 if dataset_name in binary_datasets else len(np.unique(dataset.y_train))
        )
        disc_model = instantiate(
            cfg.disc_model.model,
            input_size=dataset.X_train.shape[1],
            target_size=num_classes,
        )
        train_dataloader = dataset.train_dataloader(
            batch_size=cfg.disc_model.batch_size, shuffle=True, noise_lvl=0
        )
        test_dataloader = dataset.test_dataloader(
            batch_size=cfg.disc_model.batch_size, shuffle=False
        )
        # disc_model.load(disc_model_path)
        disc_model.fit(
            train_dataloader,
            test_dataloader,
            epochs=cfg.disc_model.epochs,
            lr=cfg.disc_model.lr,
            patience=cfg.disc_model.patience,
            checkpoint_path=disc_model_path,
        )
        disc_model.save(disc_model_path)
        logger.info("Evaluating discriminator model")
        print(classification_report(dataset.y_test, disc_model.predict(dataset.X_test)))
        report = classification_report(
            dataset.y_test, disc_model.predict(dataset.X_test), output_dict=True
        )
        pd.DataFrame(report).transpose().to_csv(
            os.path.join(save_folder, f"eval_disc_model_{fold_n}_{disc_model_name}.csv")
        )
        # run[f"{fold_n}/metrics"] = process_classification_report(
        #     report, prefix="disc_test"
        # )

        run[f"{fold_n}/disc_model"].upload(disc_model_path)

        if cfg.experiment.relabel_with_disc_model:
            dataset.y_train = disc_model.predict(dataset.X_train).detach().numpy()
            dataset.y_test = disc_model.predict(dataset.X_test).detach().numpy()

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
        # gen_model.load(gen_model_path)
        gen_model.fit(
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            num_epochs=cfg.gen_model.epochs,
            patience=cfg.gen_model.patience,
            learning_rate=cfg.gen_model.lr,
            checkpoint_path=gen_model_path,
            # neptune_run=run,
        )
        # run[f"{fold_n}/metrics/gen_model_train_time"] = time() - time_start
        gen_model.save(gen_model_path)
        run[f"{fold_n}/gen_model"].upload(gen_model_path)

        train_ll = gen_model.predict_log_prob(train_dataloader).mean().item()
        test_ll = gen_model.predict_log_prob(test_dataloader).mean().item()
        pd.DataFrame({"train_ll": [train_ll], "test_ll": [test_ll]}).to_csv(
            os.path.join(save_folder, f"eval_gen_model_{fold_n}_{gen_model_name}.csv")
        )

        logger.info("Handling counterfactual generation")

        dataset.X_test = dataset.X_test[dataset.y_test == 0]
        dataset.y_test = dataset.y_test[dataset.y_test == 0]
        cf = RPPCEF(
            K=cfg.counterfactuals.K,
            gen_model=gen_model,
            disc_model=disc_model,
            disc_model_criterion=instantiate(cfg.counterfactuals.disc_loss),
            neptune_run=run,
        )
        train_dataloader_for_log_prob = dataset.train_dataloader(
            batch_size=cfg.counterfactuals.batch_size, shuffle=False
        )
        median_log_prob = torch.median(
            gen_model.predict_log_prob(train_dataloader_for_log_prob)
        )
        run[f"{fold_n}/parameters/median_log_prob"] = median_log_prob
        print(median_log_prob)

        cf_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(dataset.X_test).float(),
                disc_model.predict(dataset.X_test).long(),
            ),
            batch_size=cfg.counterfactuals.batch_size,
            shuffle=False,
        )
        time_start = time()
        deltas, Xs, ys_orig, ys_target, _ = cf.search_batch(
            dataloader=cf_dataloader,
            epochs=cfg.counterfactuals.epochs,
            lr=cfg.counterfactuals.lr,
            patience=cfg.counterfactuals.patience,
            alpha=cfg.counterfactuals.alpha,
            beta=cfg.counterfactuals.beta,
            median_log_prob=median_log_prob,
        )
        cf_search_time = np.mean(time() - time_start)
        run[f"{fold_n}/metrics/cf_search_time"] = cf_search_time
        counterfactuals_path = os.path.join(
            save_folder, f"counterfactuals_{disc_model_name}_{fold_n}.csv"
        )
        Xs_cfs = Xs + deltas[0]().detach().numpy()
        pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
        run[f"{fold_n}/counterfactuals"].upload(counterfactuals_path)

        # Xs_cfs = pd.read_csv(counterfactuals_path).values.astype(np.float32)
        # model_returned = ~np.isnan(Xs_cfs[:, 0])
        # cf_search_time = pd.read_csv(
        #     os.path.join(save_folder, f"metrics_{disc_model_name}_cv.csv")
        # )["time"].iloc[fold_n]

        model_returned = np.ones(Xs_cfs.shape[0]).astype(bool)

        logger.info("Calculating metrics")

        metrics = evaluate_cf(
            gen_model=gen_model,
            disc_model=disc_model,
            X_cf=Xs_cfs,
            model_returned=model_returned,
            categorical_features=dataset.categorical_features,
            continuous_features=dataset.numerical_features,
            X_train=dataset.X_train,
            y_train=dataset.y_train.reshape(-1),
            X_test=Xs,
            y_test=ys_orig,
            median_log_prob=median_log_prob,
        )
        print(metrics)
        run[f"{fold_n}/metrics/cf"] = stringify_unsupported(metrics)

        metrics["time"] = cf_search_time

        log_df = pd.concat([log_df, pd.DataFrame(metrics, index=[fold_n])])
    logger.info("Finalizing and stopping run")

    log_df.to_csv(
        os.path.join(save_folder, f"metrics_{disc_model_name}_cv.csv"), index=False
    )
    run["metrics"].upload(
        os.path.join(output_folder, f"metrics_{disc_model_name}_cv.csv")
    )
    run.stop()


if __name__ == "__main__":
    main()
