import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

def analyze_dataset(dataset_path, output_analysis_folder):
    for script_type in os.listdir(dataset_path):
        script_type_path = os.path.join(dataset_path, script_type)
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
            manuscripts_per_year = {}
            pages_per_year = {}

            # Iterate over each year folder
            for year_folder in os.listdir(subtype_path):
                year_folder_path = os.path.join(subtype_path, year_folder)
                if not os.path.isdir(year_folder_path):
                    continue

                images = [img for img in os.listdir(year_folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
                if not images:
                    continue  # Skip folders with no images

                years.append(int(year_folder))  # Collect year

                # Collect manuscripts and pages data
                manuscript_ids = set()
                page_counts_in_year = 0
                for image_file in images:
                    parts = image_file.split('_')
                    manuscript_id = parts[0].split('-')[0]  # Extracting manuscript number
                    manuscript_ids.add(manuscript_id)
                    page_counts_in_year += 1

                # Update manuscript and page statistics
                manuscripts_per_year[year_folder] = len(manuscript_ids)
                pages_per_year[year_folder] = page_counts_in_year
                manuscript_count += len(manuscript_ids)
                page_count += page_counts_in_year

            if not years:
                print(f"  Skipping subtype {subtype} for {script_type}: no valid years found.")
                continue  # Skip this subtype if there are no valid year folders

            # Calculate statistics
            min_year = min(years)
            max_year = max(years)
            avg_pages_per_year = page_count / len(years)
            avg_manuscripts_per_year = manuscript_count / len(years)

            # Create the output directory for analysis
            analysis_output_path = os.path.join(output_analysis_folder, script_type, subtype)
            os.makedirs(analysis_output_path, exist_ok=True)

            # Save the analysis to a text file
            analysis_txt_path = os.path.join(analysis_output_path, 'DataAnalysisTxt.file')
            with open(analysis_txt_path, 'w') as f:
                f.write(f"Sub-dataset: {subtype} ({script_type})\n")
                f.write("=" * 40 + "\n")
                f.write(f"Number of years: {len(years)}\n")
                f.write(f"Min year: {min_year}\n")
                f.write(f"Max year: {max_year}\n")
                f.write(f"Total manuscripts: {manuscript_count}\n")
                f.write(f"Total pages: {page_count}\n")
                f.write(f"Average manuscripts per year: {avg_manuscripts_per_year:.2f}\n")
                f.write(f"Average pages per year: {avg_pages_per_year:.2f}\n")
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
            histogram_image_path = os.path.join(analysis_output_path, 'Historgram_image.jpg')
            plt.tight_layout()
            plt.savefig(histogram_image_path)
            plt.close()

            print(f"  Analysis for {subtype} saved at {analysis_output_path}")

# Example usage
dataset_path = "../../data/dataset_pages_annotated"  # Replace with your actual dataset path
output_analysis_folder = "../../Dataset_Analysis"  # Path where the analysis will be saved
analyze_dataset(dataset_path, output_analysis_folder)
