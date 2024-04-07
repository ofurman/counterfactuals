import torch
import os

import hydra
from omegaconf import DictConfig

import counterfactuals.cf_methods.diffemorphic_explanations.adv as adv
import counterfactuals.cf_methods.diffemorphic_explanations.classifiers.cnn as classifiers
import counterfactuals.cf_methods.diffemorphic_explanations.classifiers.unet as unet
from counterfactuals.cf_methods.diffemorphic_explanations.utils import load_checkpoint
from counterfactuals.cf_methods.diffemorphic_explanations.data import get_data_info
from counterfactuals.cf_methods.diffemorphic_explanations.generative_models.factory import get_generative_model

@hydra.main(
    config_path="../conf/other_methods", config_name="config_diffemorphic", version_base="1.2"
)
def main(config: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # data_set
    print("-" * 30)
    print(f"DATASET: {config.dataset.name}")
    data_info = get_data_info(config.dataset.name)

    # classifier
    print("-" * 30)
    print("CLASSIFIER")

    data_set = data_info["data_set"]

    if data_set == "Mall":
        c_type = "Mall_UNet"
        kwargs = {'unet_type': config.disc_model.unet_type}
        classifier = getattr(unet, c_type)(**kwargs)
    else:
        c_type = data_set + "_CNN"
        classifier = getattr(classifiers, c_type)()

    if config.disc_model.classifier_path is None:
        config.disc_model.classifier_path = f"checkpoints/classifiers/{c_type}.pth"
        os.makedirs(os.path.dirname(config.disc_model.classifier_path), exist_ok=True)

    _, _, _ = load_checkpoint(config.disc_model.classifier_path, classifier, device)

    classifier.to(device)
    classifier.eval()

    #generative model
    generative_model, g_model_type = get_generative_model(config.gen_model.g_type, data_info, device=device)
    data_set = data_info["data_set"]

    print("-" * 30)
    print(f"GENERATIVE MODEL: {config.gen_model.g_type}")
    if not config.gen_model.gen_path:
        config.gen_model.gen_path = f"checkpoints/generative_models/{data_set}_{g_model_type}.pth"

    _, _, _ = load_checkpoint(config.gen_model.gen_path, generative_model, device)

    generative_model.to(device)

    # adv-attac
    print("-" * 30)
    c_model = classifier
    c_model.eval()

    if config.adv_attack.attack_style == "z":
        g_model = generative_model
        g_model.eval()
    else:
        g_model = None

    adv.adv_attack(g_model, c_model, device,
                                   config.adv_attack.attack_style, data_info, config.adv_attack.num_steps, config.adv_attack.lr, config.adv_attack.save_at,
                                   config.adv_attack.target_class, config.adv_attack.image_path, config.adv_attack.result_dir, config.adv_attack.maximize)


if __name__ == '__main__':
    main()