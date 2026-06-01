import os
import pickle
from tqdm import tqdm
import torch

from .image_anonymization import run_image_swap_pipeline


def prepare_dataset(dataset_path, new_dataset_path):
    os.makedirs(new_dataset_path, exist_ok=True)

    all_items = [f for f in os.listdir(dataset_path) if f != ".DS_Store"]
    image_list = []

    if any(
        os.path.isdir(os.path.join(dataset_path, f)) for f in all_items
    ):  # folder dataset with folders inside
        fldr_list = sorted(all_items, key=lambda f: int(f))

        for fldr in fldr_list:
            source_fldr = os.path.join(dataset_path, fldr)
            target_fldr = os.path.join(new_dataset_path, fldr)
            os.makedirs(target_fldr, exist_ok=True)

            # Process files within each subfolder
            files = sorted(os.listdir(source_fldr), key=lambda f: int(f.split(".")[0]))
            for file in files:
                image_list.append(
                    (os.path.join(source_fldr, file), os.path.join(target_fldr, file))
                )
    else:  # one folder dataset with images inside
        files = sorted(all_items, key=lambda f: int(f.split(".")[0]))
        for file in files:
            image_list.append(
                (os.path.join(dataset_path, file), os.path.join(new_dataset_path, file))
            )

    return image_list


if __name__ == "__main__":
    generation_method = "single_id"
    dataset_folder = "/igd/a1/home/demiroer/IDAnonymizationPraktikum/datasets/FFHQ_toy_dataset_folder"
    dataset_name = "FFHQ_5_img"
    dataset_path = os.path.join(dataset_folder, dataset_name)
    new_dataset_path = os.path.join(
        dataset_folder, f"{dataset_name}_{generation_method}"
    )
    image_list = prepare_dataset(dataset_path, new_dataset_path)

    with open(os.path.join(dataset_folder, "caption_to_images.pkl"), "rb") as f:
        dataset_dict = pickle.load(f)

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
                dataset_dict=dataset_dict,
                number_of_ids=3,
                # this part used for ID conditioning together with IP ADAPTER
                #
                #
                # this part used for parallel generation
                n_coeff_init=-3,
                p_coeff_init=4,
                n_coeff_update_fn=lambda c, t: c,
                p_coeff_update_fn=lambda c, t: c,
                # this part used for parallel generation
                #
                #
                # diff step parameters
                inversion_num_inference_steps=50,
                inversion_guidance_scale=1.25,
                text_generation_num_inference_steps=100,
                text_generation_guidance_scale=3,
                ip_adapter_num_inference_steps=50,
                ip_adapter_guidance_scale=5.5,
                generation_method=generation_method,
            )
