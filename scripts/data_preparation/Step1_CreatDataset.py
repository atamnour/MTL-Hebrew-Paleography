import os
import shutil


def create_subtype_folders(original_dataset_path, new_dataset_path):
    # Iterate over each script type (e.g., Ashkenazi, Byzantine)
    for script_type in os.listdir(original_dataset_path):
        script_type_path = os.path.join(original_dataset_path, script_type)
        if not os.path.isdir(script_type_path):
            continue

        print(f"Processing script type: {script_type}")

        # Create new folders for Square, Semi-cursive, and Cursive inside each script type
        square_path = os.path.join(new_dataset_path, script_type, 'Square')
        semi_cursive_path = os.path.join(new_dataset_path, script_type, 'Semi-cursive')
        cursive_path = os.path.join(new_dataset_path, script_type, 'Cursive')

        os.makedirs(square_path, exist_ok=True)
        os.makedirs(semi_cursive_path, exist_ok=True)
        os.makedirs(cursive_path, exist_ok=True)

        # Iterate over each year folder in the script type
        for year_folder in os.listdir(script_type_path):
            year_folder_path = os.path.join(script_type_path, year_folder)
            if not os.path.isdir(year_folder_path):
                continue

            print(f"  Processing year folder: {year_folder}")

            # For each image in the year folder
            for image_file in os.listdir(year_folder_path):
                if image_file.endswith(('.png', '.jpg', '.jpeg')):
                    # Use case-insensitive matching for subtypes
                    lower_image_file = image_file.lower()

                    if 'square' in lower_image_file:
                        # Copy to the Square folder in the new dataset
                        new_year_folder = os.path.join(square_path, year_folder)
                        os.makedirs(new_year_folder, exist_ok=True)
                        shutil.copy(os.path.join(year_folder_path, image_file), os.path.join(new_year_folder, image_file))
                        print(f"    Copying {image_file} to Square")

                    elif 'semi-cursive' in lower_image_file:
                        # Copy to the Semi-cursive folder in the new dataset
                        new_year_folder = os.path.join(semi_cursive_path, year_folder)
                        os.makedirs(new_year_folder, exist_ok=True)
                        shutil.copy(os.path.join(year_folder_path, image_file), os.path.join(new_year_folder, image_file))
                        print(f"    Copying {image_file} to Semi-cursive")

                    elif 'cursive' in lower_image_file:
                        # Copy to the Cursive folder in the new dataset
                        new_year_folder = os.path.join(cursive_path, year_folder)
                        os.makedirs(new_year_folder, exist_ok=True)
                        shutil.copy(os.path.join(year_folder_path, image_file), os.path.join(new_year_folder, image_file))
                        print(f"    Copying {image_file} to Cursive")
                    else:
                        print(f"    Skipping {image_file}: Subtype not identified")

    print("Dataset restructuring completed.")
# Example usage
original_dataset_path = r"C:\DeepLearning\MTL-Hebrew-Paleography\data\original"
new_dataset_path = "../../data/dataset_pages_annotated"  # Path to the new dataset
create_subtype_folders(original_dataset_path, new_dataset_path)
