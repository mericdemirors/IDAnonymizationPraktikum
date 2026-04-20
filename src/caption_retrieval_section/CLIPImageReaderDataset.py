import os
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize,
    Lambda,
    InterpolationMode,
)


class CLIPImageReaderDataset(Dataset):
    def __init__(self, dataset_folder: str = None):
        # image pre-processing to fit the CLIP input expectations
        self.transform = Compose(
            [
                Resize(
                    size=224, interpolation=InterpolationMode.BICUBIC, antialias=True
                ),
                Lambda(lambda img: img.convert("RGB")),
                ToTensor(),
                Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        # we may initialize the dataset for processing a single image with a given path
        # so we handle dummy initializytion like this
        if dataset_folder is None:
            return

        # get ID folders
        fldr_list = sorted(
            [fldr for fldr in os.listdir(dataset_folder) if not fldr.startswith(".")],
            key=lambda fldr: int(fldr),
        )

        # get images under the ID folders
        self.images = []
        valid_exts = (".jpg", ".jpeg", ".png")
        for fldr in tqdm(fldr_list):
            fldr_path = os.path.join(dataset_folder, fldr)

            files = sorted(
                [f for f in os.listdir(fldr_path) if f.lower().endswith(valid_exts)],
                key=lambda f: int(f.split(".")[0]),
            )

            for file in files:
                self.images.append(os.path.join(fldr_path, file))

    def __len__(self):
        return len(self.images)

    def read_single_img(self, img_full_path: str):
        img = Image.open(img_full_path)
        img = self.transform(img)
        return img

    def __getitem__(self, idx: int):
        img_full_path = self.images[idx]

        id_folder, img_file_name = os.path.split(img_full_path)
        id_idx = os.path.split(id_folder)[1]

        img = self.read_single_img(img_full_path)

        return id_idx, img_file_name, img
