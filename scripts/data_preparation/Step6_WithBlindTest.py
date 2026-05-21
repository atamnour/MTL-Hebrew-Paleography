import os
import json
import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def split_dataset_with_blind_test(dataset_json_path, output_json_path, train_ratio=0.8, test_ratio=0.1, blind_ratio=0.1, debug=False):
    print(f"Loading dataset from: {dataset_json_path}")
    with open(dataset_json_path, 'r') as f:
        dataset = json.load(f)

    train_data = defaultdict(list)
    test_data = defaultdict(list)
    blind_test_data = defaultdict(list)
    low_patch_folders = []
    total_train_patches = 0
    error_log = []

    # Step 1: Handle folders with less than 50 patches
    small_patch_threshold = 50
    remaining_folders = {}
    for decade, patches in dataset.items():
        if len(patches) < small_patch_threshold:
            blind_test_data[decade] = patches
            low_patch_folders.append(decade)
        else:
            remaining_folders[decade] = patches

    print(f"Folders with less than {small_patch_threshold} patches added to blind test: {low_patch_folders}")

    # Step 2: Divide the remaining decades into bins for sparse blind test selection
    total_folders = len(remaining_folders)
    blind_test_count = max(1, int(total_folders * blind_ratio))
    bin_size = max(1, total_folders // blind_test_count)
    print(f"Total folders after filtering: {total_folders}, Blind test folders: {blind_test_count}, Bin size: {bin_size}")

    sorted_decades = sorted(remaining_folders.items(), key=lambda x: len(x[1]))
    decade_names = [item[0] for item in sorted_decades]
    additional_blind_test_folders = []

    for i in range(0, total_folders, bin_size):
        bin_folders = decade_names[i:i + bin_size]
        selected_folder = min(bin_folders, key=lambda x: len(remaining_folders[x]))
        additional_blind_test_folders.append(selected_folder)

    print(f"Additional blind test folders selected: {additional_blind_test_folders}")

    for decade in additional_blind_test_folders:
        blind_test_data[decade] = remaining_folders.pop(decade)

    # Step 3: Split the remaining folders into train/test
    for decade, patches in remaining_folders.items():
        random.shuffle(patches)
        train_count = int(len(patches) * train_ratio)
        train_data[decade] = patches[:train_count]
        test_data[decade] = patches[train_count:]

        total_train_patches += len(train_data[decade])

    # Step 4: Balance train data
    avg_patches_per_class = total_train_patches // len(train_data) if train_data else 0
    balanced_train_data = balance_train_dataset(train_data, avg_patches_per_class, debug)

    # Step 5: Save results
    balanced_dataset = {
        "train": balanced_train_data,
        "test": test_data,
        "blind_test": blind_test_data
    }
    with open(output_json_path, 'w') as f:
        json.dump(balanced_dataset, f, indent=4)
    print(f"Balanced dataset with blind test saved to {output_json_path}")

    # Error logging
    if error_log:
        error_log_path = os.path.join(os.path.dirname(output_json_path), 'error.txt')
        with open(error_log_path, 'w') as f:
            f.write("\n".join(error_log))
        print(f"Error log saved as '{error_log_path}'.")

    # Visualize patch distribution
    visualize_patch_distribution(balanced_train_data, test_data, blind_test_data, output_json_path)


def balance_train_dataset(train_data, avg_patches_per_class, debug=False):
    balanced_data = defaultdict(list)

    for decade, patches in train_data.items():
        patch_count = len(patches)

        if patch_count < avg_patches_per_class:
            extra_patches = random.choices(patches, k=avg_patches_per_class - patch_count)
            balanced_data[decade] = patches + extra_patches
            if debug:
                print(f"\nBalancing decade: {decade}")
                print(f"  Current patches: {patch_count}")
                print(f"  Duplicating patches to reach average of {avg_patches_per_class}")
        elif patch_count > avg_patches_per_class:
            balanced_data[decade] = random.sample(patches, avg_patches_per_class)
            if debug:
                print(f"\nBalancing decade: {decade}")
                print(f"  Current patches: {patch_count}")
                print(f"  Downsampling to reach average of {avg_patches_per_class}")
        else:
            balanced_data[decade] = patches

    return balanced_data


def visualize_patch_distribution(train_data, test_data, blind_test_data, output_path):
    decades = sorted(set(train_data.keys()) | set(test_data.keys()) | set(blind_test_data.keys()))
    train_patch_counts = [len(train_data.get(decade, [])) for decade in decades]
    test_patch_counts = [len(test_data.get(decade, [])) for decade in decades]
    blind_test_patch_counts = [len(blind_test_data.get(decade, [])) for decade in decades]

    plt.figure(figsize=(12, 6))
    plt.bar(decades, train_patch_counts, color='blue', label='Train')
    plt.bar(decades, test_patch_counts, color='red', label='Test', bottom=train_patch_counts)
    plt.bar(decades, blind_test_patch_counts, color='green', label='Blind Test', bottom=np.add(train_patch_counts, test_patch_counts))
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Decade')
    plt.ylabel('Number of Patches')
    plt.title('Patches Distribution per Decade')
    plt.legend()
    plt.tight_layout()
    distribution_plot_path = output_path.replace('.json', '_patches_distribution.jpg')
    plt.savefig(distribution_plot_path)
    plt.close()

    print(f"Patch distribution visualization saved as '{distribution_plot_path}'.")


def process_all_subtypes(root_path, train_ratio=0.8, test_ratio=0.1, blind_ratio=0.1, debug=False):
    for subtype in ["Square", "Cursive", "Semi-cursive"]:
        for root, dirs, files in os.walk(root_path):
            if subtype in root:
                json_files = [f for f in files if f.endswith('.json')]
                for json_file in json_files:
                    dataset_json_path = os.path.join(root, json_file)
                    output_json_path = os.path.join(root, f"{subtype}_balanced_split.json")
                    print(f"\nProcessing {dataset_json_path}")
                    split_dataset_with_blind_test(dataset_json_path, output_json_path, train_ratio, test_ratio, blind_ratio, debug)


# Example usage
root_path = r"/cs_storage/atamni/ServerBGU/Hebrew Paleography/Dataset/HebrewDatasetBalanced/"
process_all_subtypes(root_path, train_ratio=0.8, test_ratio=0.1, blind_ratio=0.1, debug=True)

#
#
# import os
# import json
# import random
# from collections import defaultdict
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def split_dataset_with_blind_test(dataset_json_path, output_json_path, train_ratio=0.8, test_ratio=0.1, blind_ratio=0.1, debug=False):
#     print(f"Loading dataset from: {dataset_json_path}")
#     with open(dataset_json_path, 'r') as f:
#         dataset = json.load(f)
#
#     train_data = defaultdict(list)
#     test_data = defaultdict(list)
#     blind_test_data = defaultdict(list)
#     total_train_patches = 0
#     error_log = []
#
#     # Step 1: Divide the decades into bins for sparse blind test selection
#     total_folders = len(dataset)
#     blind_test_count = max(1, int(total_folders * blind_ratio))
#     bin_size = max(1, total_folders // blind_test_count)
#     print(f"Total folders: {total_folders}, Blind test folders: {blind_test_count}, Bin size: {bin_size}")
#
#     sorted_decades = sorted(dataset.items(), key=lambda x: len(x[1]))
#     decade_names = [item[0] for item in sorted_decades]
#     blind_test_folders = []
#
#     for i in range(0, total_folders, bin_size):
#         bin_folders = decade_names[i:i + bin_size]
#         selected_folder = min(bin_folders, key=lambda x: len(dataset[x]))
#         blind_test_folders.append(selected_folder)
#
#     print(f"Blind test folders selected: {blind_test_folders}")
#
#     for decade in blind_test_folders:
#         blind_test_data[decade] = dataset[decade]
#
#     # Step 2: Split remaining folders into train/test
#     remaining_folders = {k: v for k, v in dataset.items() if k not in blind_test_folders}
#
#     for decade, patches in remaining_folders.items():
#         random.shuffle(patches)
#         train_count = int(len(patches) * train_ratio)
#         train_data[decade] = patches[:train_count]
#         test_data[decade] = patches[train_count:]
#
#         total_train_patches += len(train_data[decade])
#
#     # Step 3: Balance train data
#     avg_patches_per_class = total_train_patches // len(train_data) if train_data else 0
#     balanced_train_data = balance_train_dataset(train_data, avg_patches_per_class, debug)
#
#     # Step 4: Save results
#     balanced_dataset = {
#         "train": balanced_train_data,
#         "test": test_data,
#         "blind_test": blind_test_data
#     }
#     with open(output_json_path, 'w') as f:
#         json.dump(balanced_dataset, f, indent=4)
#     print(f"Balanced dataset with blind test saved to {output_json_path}")
#
#     # Error logging
#     if error_log:
#         error_log_path = os.path.join(os.path.dirname(output_json_path), 'error.txt')
#         with open(error_log_path, 'w') as f:
#             f.write("\n".join(error_log))
#         print(f"Error log saved as '{error_log_path}'.")
#
#     # Visualize patch distribution
#     visualize_patch_distribution(balanced_train_data, test_data, blind_test_data, output_json_path)
#
#
# def balance_train_dataset(train_data, avg_patches_per_class, debug=False):
#     balanced_data = defaultdict(list)
#
#     for decade, patches in train_data.items():
#         patch_count = len(patches)
#
#         if patch_count < avg_patches_per_class:
#             extra_patches = random.choices(patches, k=avg_patches_per_class - patch_count)
#             balanced_data[decade] = patches + extra_patches
#             if debug:
#                 print(f"\nBalancing decade: {decade}")
#                 print(f"  Current patches: {patch_count}")
#                 print(f"  Duplicating patches to reach average of {avg_patches_per_class}")
#         elif patch_count > avg_patches_per_class:
#             balanced_data[decade] = random.sample(patches, avg_patches_per_class)
#             if debug:
#                 print(f"\nBalancing decade: {decade}")
#                 print(f"  Current patches: {patch_count}")
#                 print(f"  Downsampling to reach average of {avg_patches_per_class}")
#         else:
#             balanced_data[decade] = patches
#
#     return balanced_data
#
#
# def visualize_patch_distribution(train_data, test_data, blind_test_data, output_path):
#     decades = sorted(set(train_data.keys()) | set(test_data.keys()) | set(blind_test_data.keys()))
#     train_patch_counts = [len(train_data.get(decade, [])) for decade in decades]
#     test_patch_counts = [len(test_data.get(decade, [])) for decade in decades]
#     blind_test_patch_counts = [len(blind_test_data.get(decade, [])) for decade in decades]
#
#     plt.figure(figsize=(12, 6))
#     plt.bar(decades, train_patch_counts, color='blue', label='Train')
#     plt.bar(decades, test_patch_counts, color='red', label='Test', bottom=train_patch_counts)
#     plt.bar(decades, blind_test_patch_counts, color='green', label='Blind Test', bottom=np.add(train_patch_counts, test_patch_counts))
#     plt.xticks(rotation=45, ha='right')
#     plt.xlabel('Decade')
#     plt.ylabel('Number of Patches')
#     plt.title('Patches Distribution per Decade')
#     plt.legend()
#     plt.tight_layout()
#     distribution_plot_path = output_path.replace('.json', '_patches_distribution.jpg')
#     plt.savefig(distribution_plot_path)
#     plt.close()
#
#     print(f"Patch distribution visualization saved as '{distribution_plot_path}'.")
#
#
# def process_all_subtypes(root_path, train_ratio=0.8, test_ratio=0.1, blind_ratio=0.1, debug=False):
#     for subtype in ["Square", "Cursive", "Semi-cursive"]:
#         for root, dirs, files in os.walk(root_path):
#             if subtype in root:
#                 json_files = [f for f in files if f.endswith('.json')]
#                 for json_file in json_files:
#                     dataset_json_path = os.path.join(root, json_file)
#                     output_json_path = os.path.join(root, f"{subtype}_balanced_split.json")
#                     print(f"\nProcessing {dataset_json_path}")
#                     split_dataset_with_blind_test(dataset_json_path, output_json_path, train_ratio, test_ratio, blind_ratio, debug)
#
#
# # Example usage
# root_path = r"D:\Projects\AutomaticDateEstimation\Dataset\HebrewDatasetClassification"
# process_all_subtypes(root_path, train_ratio=0.8, test_ratio=0.1, blind_ratio=0.1, debug=True)
