import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def combine_years_to_decades_for_script(
        dataset_path,
        script_type,
        output_folder,
        decade_span=10,
        initial_year_square=1420,
        initial_year_semi_cursive=1420,
        initial_year_cursive=1420,
        paths_flag=False):
    """
    Combines years into decades for a specific script type and generates statistics, JSON files, and histograms.

    :param dataset_path: Path to the dataset.
    :param script_type: The script type to process (e.g., Ashkenazi, Byzantine, etc.).
    :param output_folder: Folder where the output files (JSON, text, histogram) will be saved.
    :param decade_span: Number of years in each decade/class (default is 10).
    :param initial_year_square: Initial year for the 'Square' subtype.
    :param initial_year_semi_cursive: Initial year for the 'Semi-cursive' subtype.
    :param initial_year_cursive: Initial year for the 'Cursive' subtype.
    :param paths_flag: If True, include full patch paths instead of just the filenames.
    """
    script_type_path = os.path.join(dataset_path, script_type)
    if not os.path.isdir(script_type_path):
        print(f"Script type {script_type} not found in dataset.")
        return

    print(f"Processing script type: {script_type}")

    # Ensure the output folder exists
    output_folder = os.path.join(output_folder, script_type)
    os.makedirs(output_folder, exist_ok=True)

    # Define initial years per subtype
    initial_years = {
        'Square': initial_year_square,
        'Semi-cursive': initial_year_semi_cursive,
        'Cursive': initial_year_cursive
    }

    for subtype in os.listdir(script_type_path):
        subtype_path = os.path.join(script_type_path, subtype)
        if not os.path.isdir(subtype_path):
            continue

        if subtype not in initial_years:
            print(f"  Skipping unrecognized subtype: {subtype}")
            continue

        print(f"  Processing subtype: {subtype}")

        initial_year = initial_years[subtype]

        combined_dataset = defaultdict(list)
        manuscript_per_decade = defaultdict(list)
        patches_per_decade = defaultdict(int)
        total_manuscripts = 0
        total_patches = 0
        years_in_subtype = set()

        # Process each year folder
        for year_folder in os.listdir(subtype_path):
            year_folder_path = os.path.join(subtype_path, year_folder)
            if not os.path.isdir(year_folder_path):
                continue

            year = int(year_folder)
            years_in_subtype.add(year)

            decade_start = (year - initial_year) // decade_span * decade_span + initial_year
            decade_end = decade_start + decade_span
            if decade_span == 1:
                class_name = f"{decade_start}"
            else:
                class_name = f"{decade_start}_{decade_end}"

            manuscript_ids = set()
            # Collect manuscript IDs and count patches
            patches_count = 0

            for image_file in os.listdir(year_folder_path):
                if image_file.endswith(('.png', '.jpg', '.jpeg')):
                    manuscript_id = image_file.split('_')[0].split('-')[0]
                    manuscript_ids.add(manuscript_id)
                    patches_count += 1  # Count each patch (image)

                    if paths_flag:
                        # Add the full path of the patch image
                        full_path = os.path.join(year_folder_path, image_file)
                        combined_dataset[class_name].append(full_path)

            # Record manuscripts and patch counts
            if not paths_flag:
                for manuscript_id in manuscript_ids:
                    combined_dataset[class_name].append(f"{year}-{manuscript_id}")
            manuscript_per_decade[class_name].append(len(manuscript_ids))
            patches_per_decade[class_name] += patches_count
            total_patches += patches_count
            total_manuscripts += len(manuscript_ids)

        # After processing, iterate and rename the classes by adding index i
        final_combined_dataset = {}
        for i, (class_name, manuscripts) in enumerate(combined_dataset.items()):
            if decade_span == 1:
                new_class_name = f"{class_name}"  # Add index i to the class name
            else:
                new_class_name = f"class_{i}-{class_name}"  # Add index i to the class name

          #  new_class_name = f"class_{i}-{class_name}"  # Add index i to the class name
            final_combined_dataset[new_class_name] = manuscripts

        # Save the combined data to JSON file
        json_output_path = os.path.join(output_folder, subtype, f'{subtype}_decade_combination.json')
        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

        with open(json_output_path, 'w') as json_file:
            json.dump(final_combined_dataset, json_file, indent=4)
        print(f"  Combined decade data saved to {json_output_path}")

        # Calculate average patches per decade
        avg_patches_per_decade = np.mean(list(patches_per_decade.values())) if patches_per_decade else 0

        # Save statistics to text file
        txt_output_path = os.path.join(output_folder, subtype, f'{subtype}_decade_statistics.txt')
        total_decades = len(final_combined_dataset)

        with open(txt_output_path, 'w') as txt_file:
            txt_file.write(f"Decade Combination Statistics for {subtype} ({script_type})\n")
            txt_file.write("=" * 40 + "\n")
            txt_file.write(f"Number of classes (decades): {total_decades}\n")
            txt_file.write(f"Number of years: {len(years_in_subtype)}\n")
            txt_file.write(f"Total manuscripts: {total_manuscripts}\n")
            txt_file.write(f"Total patches (images): {total_patches}\n")
            txt_file.write(
                f"Average manuscripts per decade: {np.mean([len(manuscripts) for manuscripts in manuscript_per_decade.values()]):.2f}\n")
            txt_file.write(f"Average patches per decade: {avg_patches_per_decade:.2f}\n")
            txt_file.write("=" * 40 + "\n")

        print(f"  Statistics saved to {txt_output_path}")


        decades = sorted(manuscript_per_decade.keys())
        manuscript_counts = [sum(manuscript_per_decade[decade]) for decade in decades]
        patch_counts = [patches_per_decade[decade] for decade in decades]

        # Create and save histogram for manuscripts per decade
        manuscript_histogram_output_path = os.path.join(output_folder, subtype,
                                                        f'{subtype}_decade_manuscripts_histogram.jpg')

        plt.figure(figsize=(10, 6))
        plt.bar(decades, manuscript_counts, color='blue')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Decade')
        plt.ylabel('Number of Manuscripts')
        plt.title(f'Number of Manuscripts per Decade for {subtype} ({script_type})')
        plt.tight_layout()
        plt.savefig(manuscript_histogram_output_path)
        plt.close()

        print(f"  Manuscript histogram saved to {manuscript_histogram_output_path}")

        # Create and save histogram for patches per decade
        patch_histogram_output_path = os.path.join(output_folder, subtype, f'{subtype}_decade_patches_histogram.jpg')

        plt.figure(figsize=(10, 6))
        plt.bar(decades, patch_counts, color='green')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Decade')
        plt.ylabel('Number of Patches')
        plt.title(f'Number of Patches per Decade for {subtype} ({script_type})')
        plt.tight_layout()
        plt.savefig(patch_histogram_output_path)
        plt.close()

        print(f"  Patch histogram saved to {patch_histogram_output_path}")


# Example usage
dataset_path = r"/cs_storage/atamni/ServerBGU/AutomaticDateEstimation/HebrewDataset/HebrewPatchesDataset/"  # Replace with the actual path
output_folder = "HebrewDatasetBalanced"  # Folder where the results will be saved
script_type = "Yemenite"  # Choose the script type you want to process

# Combine years with different initial years for Square, Semi-cursive, and Cursive
combine_years_to_decades_for_script(
    dataset_path,
    script_type,
    output_folder,
    decade_span=1,
    initial_year_square=800,
    initial_year_semi_cursive=2000,
    initial_year_cursive=0,
    paths_flag=True
)


# Ashkenazi
# Byzantine
# Italian
# Oriental
# Sefardic
# Yemenite
#A 1177
#A 1227
#A 1467

#B 1127
#B 1212
#B 0

#I 1073
#I 1246
#I 0

#O 895
#O 1020
#O 0

#Y 1222
#Y 1432
#Y 0

#S 990
#S 1119
#S 1307
