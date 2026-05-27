import os
from tqdm import tqdm
import torch
from insightface.app import FaceAnalysis

from .image_anonymization import run_image_swap_pipeline
from .caption_retrieval_section.utils import load_clip_model
from .ddim_inversion_section.utils import prepare_pipe_and_scheduler


def prepare_dataset(dataset_path, new_dataset_path):
    os.makedirs(new_dataset_path, exist_ok=True)

    fldr_list = sorted(
        [fldr for fldr in os.listdir(dataset_path) if (fldr != ".DS_Store")],
        key=lambda fldr: int(fldr),
    )

    for fldr in fldr_list:
        os.makedirs(os.path.join(new_dataset_path, fldr), exist_ok=True)

    os.makedirs(new_dataset_path, exist_ok=True)
    image_list = [
        (
            os.path.join(dataset_path, fldr, file),
            os.path.join(new_dataset_path, fldr, file),
        )
        for fldr in fldr_list
        for file in sorted(
            os.listdir(os.path.join(dataset_path, fldr)),
            key=lambda file: int(file.split(".")[0]),
        )
    ]

    return image_list


if __name__ == "__main__":
    dataset_path = "/igd/a1/home/demiroer/IDAnonymizationPraktikum/datasets/toy_dataset"
    new_dataset_path = "/igd/a1/home/demiroer/IDAnonymizationPraktikum/datasets/single_id_different_race_anon_dataset"
    image_list = prepare_dataset(dataset_path, new_dataset_path)

    GPU_ID = 0
    for image_path, image_save_path in tqdm(image_list):
        with torch.no_grad():
            run_image_swap_pipeline(
                # base parameters
                image_path,
                image_save_path=image_save_path,
                config_path="src/caption_retrieval_section/clip_configs/race_age_gender_different_race_attributes_config/race_age_gender_different_race_attributes_config.py",
                clip_version="ViT-L/14",
                pipe_version="SG161222/Realistic_Vision_V4.0_noVAE",
                vae_version="stabilityai/sd-vae-ft-mse",
                device=f"cuda:{GPU_ID}",
                # base parameters
                #
                #
                # this part used for ID conditioning together with IP ADAPTER
                face_app_name="buffalo_l",
                face_app_det_size=512,
                face_app_ctx_id=GPU_ID,
                ip_adapter_ckpt="/igd/a1/home/demiroer/IDAnonymizationPraktikum/src/ip_adapter_section/ip-adapter-faceid_sd15.bin",
                ip_adapter_width=512,
                ip_adapter_height=512,
                new_id_retrieval_method="generation",
                dataset_dict={
                    "A portrait photo of a 35 year old Indian Female.": [
                        "/igd/a1/home/demiroer/IDAnonymizationPraktikum/datasets/anon_toy_dataset/4/0.png",
                        "/igd/a1/home/demiroer/IDAnonymizationPraktikum/datasets/anon_toy_dataset/4/3.png",
                        "/igd/a1/home/demiroer/IDAnonymizationPraktikum/datasets/anon_toy_dataset/4/4.png",
                    ]
                },
                number_of_ids=3,
                # this part used for ID conditioning together with IP ADAPTER
                #
                #
                # this part used for parallel generation
                n_coeff_init=0,
                p_coeff_init=1,
                n_coeff_update_fn=lambda c, t: c,
                p_coeff_update_fn=lambda c, t: c,
                # this part used for parallel generation
                #
                #
                # how to ddim inverse the image
                inversion_num_inference_steps=50,
                inversion_guidance_scale=1.25,
                generation_num_inference_steps=150,  # only for text generation
                generation_guidance_scale=5.5,  # only for text generation
                ip_adapter_num_inference_steps=50,  # only for ID generation
                ip_adapter_guidance_scale=5.5,  # only for ID generation
                generation_method="single_id",
            )
