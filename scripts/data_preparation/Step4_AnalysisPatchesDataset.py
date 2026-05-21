import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

def check_missing_pages_and_patches(original_dataset, patches_dataset, output_analysis_folder):
    """
    Check for missing pages and pages with less than 100 patches.
    Generates individual missing patches summary for each sub-dataset.
    """
    for script_type in os.listdir(original_dataset):
        script_type_path = os.path.join(original_dataset, script_type)
        if not os.path.isdir(script_type_path):
            continue

        print(f"Checking script type: {script_type}")

        # Loop through each subtype in the original dataset
        for subtype in os.listdir(script_type_path):
            subtype_path = os.path.join(script_type_path, subtype)
            if not os.path.isdir(subtype_path):
                continue

            print(f"  Checking subtype: {subtype}")

            # Find the corresponding patches folder
            patches_subtype_path = os.path.join(patches_dataset, script_type, subtype)
            if not os.path.exists(patches_subtype_path):
                print(f"  Missing patches folder for {subtype} ({script_type})")
                continue

            missing_patches_summary = []
            empty_years = []
            years_with_less_than_50_patches = []
            valid_pages = 0
            # Check each year folder
            for year_folder in os.listdir(subtype_path):
                year_folder_path = os.path.join(subtype_path, year_folder)
                if not os.path.isdir(year_folder_path):
                    continue

                patches_year_folder_path = os.path.join(patches_subtype_path, year_folder)
                original_pages = [img for img in os.listdir(year_folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
                # print(os.listdir(patches_year_folder_path))

                # # Check if the patches year folder exists
                # if not os.path.exists(patches_year_folder_path):
                #     empty_years.append(year_folder)
                #     continue
                if len(os.listdir(patches_year_folder_path)) == 0:
                    empty_years.append(year_folder)
                    continue
                patches_pages = [img for img in os.listdir(patches_year_folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
                # Check for pages with less than 50 patches
                for page in original_pages:
                    patches_for_page = [p for p in patches_pages if p.startswith(os.path.splitext(page)[0])]
                    print("page ",page)
                    print("patches for page ",patches_for_page)
                    print("patches for page ",len(patches_for_page))
                    if len(patches_for_page) < 50:
                        years_with_less_than_50_patches.append((year_folder, page, len(patches_for_page)))
                    if len(patches_for_page) <= 100 and len(patches_for_page)  >=70:
                        valid_pages = valid_pages + 1


            # Save the missing patches summary to a text file
            missing_patches_file = os.path.join(output_analysis_folder, script_type, subtype, "missing_patches_summary.txt")
            os.makedirs(os.path.dirname(missing_patches_file), exist_ok=True)
            with open(missing_patches_file, 'w') as f:
                f.write(f"Missing Patches Summary for {subtype} ({script_type})\n")
                f.write("=" * 40 + "\n")
                f.write(f"Number of empty years: {len(empty_years)} out of {len(os.listdir(subtype_path))}\n")
                f.write(f"Number of valid pages is: {valid_pages}")
                f.write(f"Number of years with less than 50 patches: {len(years_with_less_than_50_patches)} out of {len(os.listdir(subtype_path))}\n")
                f.write("\nEmpty Years:\n")
                for year in empty_years:
                    f.write(f"{year}\n")
                f.write("\nYears with less than 50 patches:\n")
                for year, page, patch_count in years_with_less_than_50_patches:
                    f.write(f"{year} (Page: {page}): {patch_count} patches\n")
            print(f"Missing patches summary saved at {missing_patches_file}")


def analyze_patches_dataset(patches_dataset, output_analysis_folder):
    """
    Analyze the patches dataset and calculate various statistics.
    """
    for script_type in os.listdir(patches_dataset):
        script_type_path = os.path.join(patches_dataset, script_type)
        if not os.path.isdir(script_type_path):
            continue

        print(f"Analyzing script type: {script_type}")

        # Analyze each subtype (Square, Semi-cursive, Cursive)
        for subtype in os.listdir(script_type_path):
            subtype_path = os.path.join(script_type_path, subtype)
            if not os.path.isdir(subtype_path):
                continue

            print(f"  Analyzing subtype: {subtype}")

            # Initialize variables for analysis
            years = []
            manuscript_count = 0
            page_count = 0
            patch_count = 0
            manuscripts_per_year = {}
            pages_per_year = {}
            total_pages_with_sufficient_patches = 0

            # Iterate over each year folder
            for year_folder in os.listdir(subtype_path):
                year_folder_path = os.path.join(subtype_path, year_folder)
                if not os.path.isdir(year_folder_path):
                    continue

                images = [img for img in os.listdir(year_folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
                if not images:
                    continue  # Skip folders with no images

                years.append(int(year_folder))  # Collect year

                # Collect manuscripts, pages, and patches data
                manuscript_ids = set()
                pages_ids = set()
                page_counts_in_year = 0
                patch_counts_in_year = len(images)
                for image_file in images:
                    parts = image_file.split('_')
                    manuscript_id = parts[0].split('-')[0]  # Extracting manuscript number
                    page_id = image_file.split('_')[0]
                    manuscript_ids.add(manuscript_id)
                    pages_ids.add(page_id)
                    page_counts_in_year += 1

                    # Check if this page has sufficient patches
                    if 70 <= patch_counts_in_year <= 100:
                        total_pages_with_sufficient_patches += 1

                # Update manuscript, page, and patch statistics
                manuscripts_per_year[year_folder] = len(manuscript_ids)
                pages_per_year[year_folder] = len(pages_ids)
                manuscript_count += len(manuscript_ids)
                page_count += len(pages_ids)
                patch_count += patch_counts_in_year

            if not years:
                print(f"  Skipping subtype {subtype} for {script_type}: no valid years found.")
                continue  # Skip this subtype if there are no valid year folders

            # Calculate statistics
            min_year = min(years)
            max_year = max(years)
            avg_pages_per_year = page_count / len(years)
            avg_manuscripts_per_year = manuscript_count / len(years)
            empty_years_count = sum(1 for year in pages_per_year.values() if year == 0)
            years_with_less_than_100_patches = sum(1 for year in pages_per_year.values() if year < 100)

            # Calculate number of years with at least 2 manuscripts
            years_with_2_or_more_manuscripts = sum(1 for year in manuscripts_per_year.values() if year >= 2)

            # Create the output directory for analysis
            analysis_output_path = os.path.join(output_analysis_folder, script_type, subtype)
            os.makedirs(analysis_output_path, exist_ok=True)

            # Save the analysis to a text file
            analysis_txt_path = os.path.join(analysis_output_path, 'DataPatchesAnalysisTxt.file')
            with open(analysis_txt_path, 'w') as f:
                f.write(f"Sub-dataset: {subtype} ({script_type})\n")
                f.write("=" * 40 + "\n")
                f.write(f"Number of years: {len(years)}\n")
                f.write(f"Min year: {min_year}\n")
                f.write(f"Max year: {max_year}\n")
                f.write(f"Total manuscripts: {manuscript_count}\n")
                f.write(f"Total pages: {page_count}\n")
                f.write(f"Total patches: {patch_count}\n")
                f.write(f"Average manuscripts per year: {avg_manuscripts_per_year:.2f}\n")
                f.write(f"Average pages per year: {avg_pages_per_year:.2f}\n")
                f.write(f"Years with 2 or more manuscripts: {years_with_2_or_more_manuscripts}\n")
                f.write(f"Empty years: {empty_years_count}\n")
                f.write(f"Years with less than 100 patches: {years_with_less_than_100_patches}\n")
                f.write(f"Total pages with 70-100 patches: {total_pages_with_sufficient_patches}/{page_count}\n")
                f.write("=" * 40 + "\n")

            # Plot and save the histogram of manuscripts and pages per year
            years_sorted = sorted(years)
            manuscripts_per_year_sorted = [manuscripts_per_year[str(year)] for year in years_sorted]
            pages_per_year_sorted = [pages_per_year[str(year)] for year in years_sorted]

            x = np.arange(len(years_sorted))
            width = 0.35  # Width of bars

            fig, ax = plt.subplots(figsize=(10, 6))

            # Bar chart for manuscripts
            ax.bar(x - width / 2, manuscripts_per_year_sorted, width, label='Manuscripts')

            # Bar chart for pages
            ax.bar(x + width / 2, pages_per_year_sorted, width, label='Pages')

            # Labeling and legend
            ax.set_xlabel('Year')
            ax.set_ylabel('Count')
            ax.set_title(f'Number of Manuscripts and Pages per Year for {subtype}')
            ax.set_xticks(x)
            ax.set_xticklabels(years_sorted, rotation=45)
            ax.legend()

            # Save the histogram image
            histogram_image_path = os.path.join(analysis_output_path, 'Historgram_patches.jpg')
            plt.tight_layout()
            plt.savefig(histogram_image_path)
            plt.close()

            print(f"  Analysis for {subtype} saved at {analysis_output_path}")

# Example usage

main_dataset = "HebrowDataset/Main_dataset_with_subtypes"  # Replace with the path to the main dataset
patches_dataset = "HebrowDataset/HebrowPatchesDataset"  # Replace with the path to the patches dataset
output_folder = "Dataset_Analysis"  # Folder where the results will be saved



# Step 1: Check for missing pages and patches
check_missing_pages_and_patches(main_dataset, patches_dataset, output_folder)

# Step 2: Analyze the patches dataset
analyze_patches_dataset(patches_dataset, output_folder)