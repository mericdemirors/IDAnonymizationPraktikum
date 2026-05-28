import os
from pathlib import Path
from PIL import Image


def stitch_image_grid(dataset_path, F, I, S, output_name="stitched_grid.png"):
    """
    Creates an F x I grid of images scaled by S and saves it next to the dataset folder.

    :param dataset_path: Path to the root dataset directory
    :param F: Number of folders (rows) to include
    :param I: Number of images (columns) per folder to include
    :param S: Scale factor. Can be a single number (e.g., 1 or 0.5) or a tuple (sx, sy)
    :param output_name: Filename for the output stitched image
    """
    dataset_dir = Path(dataset_path)

    # Helper to sort files/folders numerically if they are digits, otherwise alphabetically
    sort_key = lambda p: int(p.stem) if p.stem.isdigit() else p.stem

    # 1. Get and sort the subfolders (Rows)
    folders = sorted([d for d in dataset_dir.iterdir() if d.is_dir()], key=sort_key)
    folders = folders[:F]

    if not folders:
        raise ValueError(f"No subfolders found in {dataset_path}")

    # 2. Parse the scale factor S
    if isinstance(S, (int, float)):
        sx, sy = S, S
    elif isinstance(S, (tuple, list)) and len(S) == 2:
        sx, sy = S[0], S[1]
    else:
        raise ValueError(
            "S must be a single number or a tuple/list of two numbers (sx, sy)"
        )

    # 3. Determine base image size from the very first image
    valid_extensions = {".png", ".jpg", ".jpeg"}
    first_img_path = next(
        (
            f
            for f in sorted(folders[0].iterdir(), key=sort_key)
            if f.suffix.lower() in valid_extensions
        ),
        None,
    )

    if not first_img_path:
        raise FileNotFoundError(
            f"Could not find a valid image in the first folder: {folders[0]}"
        )

    with Image.open(first_img_path) as img:
        orig_w, orig_h = img.size

    # Calculate new scaled dimensions per cell
    new_w = int(orig_w * sx)
    new_h = int(orig_h * sy)

    # 4. Create the blank canvas for the grid
    grid_w = I * new_w
    grid_h = F * new_h
    # Using a black canvas; change to (255, 255, 255) for white if preferred
    grid_canvas = Image.new("RGB", (grid_w, grid_h), color=(0, 0, 0))

    print(f"Creating a {F}x{I} grid. Each cell scaled to {new_w}x{new_h}px.")
    print(f"Total canvas size: {grid_w}x{grid_h}px.")

    # 5. Populate the grid
    for f_idx, folder in enumerate(folders):
        # Get and sort images in the current folder (Columns)
        img_paths = sorted(
            [
                f
                for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in valid_extensions
            ],
            key=sort_key,
        )
        img_paths = img_paths[:I]

        for i_idx, img_path in enumerate(img_paths):
            try:
                with Image.open(img_path) as img:
                    # Resize individual image
                    resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                    # Calculate grid coordinates
                    x_offset = i_idx * new_w
                    y_offset = f_idx * new_h

                    # Paste into canvas
                    grid_canvas.paste(resized_img, (x_offset, y_offset))
            except Exception as e:
                print(f"Warning: Failed to process {img_path}. Skipping. Error: {e}")

    # 6. Save the final image in the parent directory (next to dataset)
    output_path = dataset_dir.parent / output_name
    grid_canvas.save(output_path)
    print(f" Done! Stitched image saved to: {output_path}")


# --- Execution block ---
if __name__ == "__main__":
    # Path exactly matching your environment setup
    DATASET_PATH = "/igd/a1/home/demiroer/IDAnonymizationPraktikum/datasets/parallel_text_p15_different_race_anon_dataset"
    OUT_NAME = f"{DATASET_PATH}_stitched.png"

    # Adjust your grid parameters here
    stitch_image_grid(
        dataset_path=DATASET_PATH,
        F=5,  # Number of folder rows
        I=5,  # Number of image columns
        S=(1 / 2, 1 / 2),  # Scale proportions (can also just pass 1 or 0.5)
        output_name=OUT_NAME,
    )
