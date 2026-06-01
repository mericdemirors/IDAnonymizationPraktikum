import os
from PIL import Image
from tqdm import tqdm


def create_image_grids(
    source_folder,
    output_folder,
    sort_by_size,
    row_count=5,
    column_count=5,
    img_w=64,
    img_h=64,
):
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Supported image extensions
    valid_extensions = (".jpg", ".jpeg", ".png")

    # Get all valid image paths from the source folder
    image_paths = sorted(
        [
            os.path.join(source_folder, f)
            for f in os.listdir(source_folder)
            if f.lower().endswith(valid_extensions)
        ],
        key=lambda f: int(os.path.basename(f).split(".")[0]),
    )

    if not image_paths:
        print(f"No valid images found in '{source_folder}'.")
        return

    print(f"Found {len(image_paths)} images. Processing grids...")

    if sort_by_size:
        image_paths.sort(key=os.path.getsize, reverse=False)

    # Calculate total images needed per grid (5x5 = 25)
    imgs_per_grid = row_count * column_count

    # Calculate the pixel dimensions of the final stitched canvas
    grid_pixel_height = row_count * img_h
    grid_pixel_width = column_count * img_w

    grid_count = 0

    # Loop through the images in chunks of 25
    for i in tqdm(range(0, len(image_paths), imgs_per_grid)):
        chunk = image_paths[i : i + imgs_per_grid]

        # Create a new blank white canvas for the grid
        # (Change (255, 255, 255) to (0, 0, 0) if you prefer a black background for partial grids)
        grid_image = Image.new(
            "RGB", (grid_pixel_width, grid_pixel_height), (255, 255, 255)
        )

        for index, img_path in enumerate(chunk):
            try:
                with Image.open(img_path) as img:
                    # Convert to RGB (handles grayscale or RGBA transparent images gracefully)
                    img = img.convert("RGB")
                    # Resize to 64x64
                    img_resized = img.resize((img_w, img_h))

                    # Calculate X and Y grid coordinates (0 to 4)
                    row_i = index // column_count
                    col_i = index % column_count

                    # Calculate actual pixel position
                    x_pos = col_i * img_w
                    y_pos = row_i * img_h

                    # Paste the image onto the canvas
                    grid_image.paste(img_resized, (x_pos, y_pos))
            except Exception as e:
                print(f"Skipping corrupt or unreadable image {img_path}: {e}")

        # Save the completed grid
        grid_count += 1
        output_path = os.path.join(output_folder, f"grid_{grid_count:03d}.jpg")
        grid_image.save(output_path, "JPEG", quality=90)
        print(f"Saved: {output_path}")

    print("Stitching complete!")


# --- Configuration ---
SOURCE_DIR = "/igd/a1/home/demiroer/IDAnonymizationPraktikum/datasets/FFHQ_toy_dataset_folder/FFHQ_5_img_single_id"
OUTPUT_DIR = (
    "/igd/a1/home/demiroer/IDAnonymizationPraktikum/datasets/FFHQ_toy_dataset_folder"
)

# Run the function
if __name__ == "__main__":
    create_image_grids(
        SOURCE_DIR,
        OUTPUT_DIR,
        sort_by_size=False,
        row_count=2,
        column_count=3,
        img_w=256,
        img_h=256,
    )
