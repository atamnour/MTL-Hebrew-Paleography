import _thread
import os
import shutil
import numpy as np
import random
import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk
from tqdm import tqdm
from scipy.ndimage import label as bwlabel
from scipy.signal import find_peaks
from skimage.measure import regionprops
import statistics
from skimage.filters import *
from scipy.signal import savgol_filter
import _thread
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
from collections import Counter
from MultiTaskModel import MultiTaskModel
from ModelsConfigsNew import modelsConfigDict

# Load category mapping
# CATEGORY_MAPPING = {
#     0:'Ashkenazi_Semi-cursive', 1: 'Ashkenazi_Cursive',
#     2: 'Byzantine_Square', 3: 'Byzantine_Semi-cursive', 4: 'Italian_Square',
#     5: 'Italian_Semi-cursive', 6: 'Oriental_Square', 7: 'Oriental_Semi-cursive',
#     8: 'Sefardic_Square', 9: 'Sefardic_Semi-cursive', 10: 'Sefardic_Cursive',
#     11: 'Yemenite_Square', 12: 'Yemenite_Semi-cursive'
# }
CATEGORY_MAPPING = {
    0: 'Ashkenazi_Square', 1: 'Ashkenazi_Semi-cursive', 2: 'Ashkenazi_Cursive',
    3: 'Byzantine_Square', 4: 'Byzantine_Semi-cursive', 5: 'Italian_Square',
    6: 'Italian_Semi-cursive', 7: 'Oriental_Square', 8: 'Oriental_Semi-cursive',
    9: 'Sefardic_Square', 10: 'Sefardic_Semi-cursive', 11: 'Sefardic_Cursive',
    12: 'Yemenite_Square', 13: 'Yemenite_Semi-cursive'
}
# Min and max year for normalization
MIN_YEAR = 895
MAX_YEAR = 1500
# Min and max decade for normalization
MIN_DECADE = 0
MAX_DECADE = 65

# Define transforms for the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

calculate_average_number_of_line_in_patch_MAX_ITETATIONS = 250  # 1000#200#10000
get_valid_patch_MAX_ITETATIONS = 250  # 1000#200#10000


def number_of_peaks(data, thresh):
    data = savgol_filter(data, 51, 5)
    peaks, _ = find_peaks(data, prominence=1, width=20, distance=20)
    return peaks.shape[0]


def is_valid_patch(patch, binary_patch, x_var_thresh=500, y_var_thresh=500, y_peaks_thresh=220, alpha=0.01, beta=0.7,
                   validate_based_on_cc=True, cc_thresh=30):
    inv_patch = ((255 - patch) / 255)
    x_profile = np.sum(inv_patch, axis=0)
    y_profile = np.sum(inv_patch, axis=1)

    x_r, x_l = np.sum(x_profile[:x_profile.shape[0] // 2]), np.sum(x_profile[x_profile.shape[0] // 2:])
    y_r, y_l = np.sum(y_profile[:y_profile.shape[0] // 2]), np.sum(y_profile[y_profile.shape[0] // 2:])

    x_var = x_profile.var()  # centered lines
    y_var = y_profile.var()  # number of lines

    f_is_valid = beta * (patch.shape[0] * patch.shape[1]) > np.count_nonzero(binary_patch) > alpha * (
                patch.shape[0] * patch.shape[1])

    lines_number = number_of_peaks(y_profile, y_peaks_thresh)

    cc_valid = True

    if validate_based_on_cc:
        contours, hierarchy = cv2.findContours(255 - patch.astype(np.uint8), 1, 2)

        boxes = []
        conts_bbs = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            conts_bbs.append(np.asarray([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=int))

            if 0.001 * patch.shape[0] * patch.shape[1] < area < 0.2 * patch.shape[0] * patch.shape[1]:
                box = np.asarray([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=int)
                boxes.append(box)

        cc_valid = len(boxes) > cc_thresh

    is_valid = f_is_valid and y_var > y_var_thresh and x_var < x_var_thresh and lines_number > 0 and \
               (1.5 > x_r / x_l > 0.5) and (1.5 > y_r / y_l > 0.5) and patch.min() != patch.max() and cc_valid

    return is_valid, lines_number


def get_patch(img, x, y, src_patch_size, dst_patch_size):
    patch = img[x:x + src_patch_size[0], y:y + src_patch_size[1]]
    patch = cv2.resize(patch, dst_patch_size)
    return patch


def get_random_patch_location(img, patch_size=(128, 128)):  # 350,350
    rows, cols = img.shape
    x = random.randint(0, rows - patch_size[0])
    y = random.randint(0, cols - patch_size[1])
    return x, y


def calculate_average_number_of_line_in_patch(img, patch_size, samples_number=20,
                                              max_iterations=calculate_average_number_of_line_in_patch_MAX_ITETATIONS):
    number_of_lines = 0
    sampled_patches = 0

    iterations = 0

    while sampled_patches < samples_number:
        x, y = get_random_patch_location(img)
        patch = get_patch(img, x, y, src_patch_size=patch_size, dst_patch_size=(350, 350))

        iterations += 1

        if iterations > max_iterations:
            return -1

        if patch.min() == patch.max():
            continue

        radius = 15
        selem = disk(radius)
        local_otsu = rank.otsu(patch, selem)
        threshold_global_otsu = threshold_otsu(patch)
        bin_patch = 255 * (patch >= threshold_global_otsu)

        is_valid, lines = is_valid_patch(bin_patch, 1 - (patch >= threshold_global_otsu), x_var_thresh=1500,
                                         y_var_thresh=500, y_peaks_thresh=80, alpha=0.01, cc_thresh=10)
        # is_valid_patch(bin_patch, 1-(patch >= threshold_global_otsu),x_var_thresh=1500, y_var_thresh=1500, y_peaks_thresh=80, alpha=class_0.1)

        if is_valid:
            sampled_patches += 1
            number_of_lines += lines

    return number_of_lines / sampled_patches


def get_valid_patch(img, src_patch_size, dst_patch_size, margine_p_x=0.1, margine_p_y=0.1,
                    max_iterations=get_valid_patch_MAX_ITETATIONS, validate_based_on_cc=True):
    iterations = 0

    x, y = None, None
    is_valid = False
    while not is_valid:
        x, y = get_random_patch_location(img)

        patch = get_patch(img, x, y, src_patch_size=src_patch_size, dst_patch_size=dst_patch_size)

        iterations += 1

        if patch.min() == patch.max():
            continue

        radius = 15
        selem = disk(radius)
        local_otsu = rank.otsu(patch, selem)
        threshold_global_otsu = threshold_otsu(patch)
        bin_patch = 255 * (patch >= threshold_global_otsu)

        is_valid, lines = is_valid_patch(bin_patch, 1 - (patch >= threshold_global_otsu), x_var_thresh=1500,
                                         y_var_thresh=500,
                                         y_peaks_thresh=80, alpha=0.01, validate_based_on_cc=validate_based_on_cc,
                                         cc_thresh=30)
        # is_valid_patch(bin_patch, 1-(patch >= threshold_global_otsu),x_var_thresh=1500, y_var_thresh=1500, y_peaks_thresh=80, alpha=class_0.1)

        if iterations > max_iterations:
            return None, -1

    return patch, [x, y]  # lines


def generate_patches_for_style(page_path, number_of_patches_per_page, patch_size,
                               number_of_lines_per_patch=5, margine_p_x=0.1, margine_p_y=0.1):
    existing_patches = 0  # Start patch count

    page_img = cv2.imread(page_path, 0)

    if (min(page_img.shape) < patch_size[0]):
        print('Error page too small', page_path)
        return

    init_patch_size = (page_img.shape[0] // 10, page_img.shape[0] // 10)

    avg_lines_num = calculate_average_number_of_line_in_patch(page_img, patch_size=init_patch_size)

    if avg_lines_num < 0:
        print('Error couldn\'t calculate avg number of line in page', page_path)
        return

    src_patch_size = (int(init_patch_size[0] * (number_of_lines_per_patch / avg_lines_num)),
                      int(init_patch_size[1] * (number_of_lines_per_patch / avg_lines_num)))

    patch_coords = []
    patches = []
    for i in range(number_of_patches_per_page):
        patch, x_y = get_valid_patch(page_img, src_patch_size=src_patch_size, dst_patch_size=patch_size,
                                     margine_p_x=margine_p_x, margine_p_y=margine_p_y)

        if patch is None:
            print(f'Error: unable to generate patches from page {page_path}')
            break
        x, y = x_y
        # x,y correspond to the TOP-LEFT in the original image
        # The bottom-right corner is (x + src_patch_size[0], y + src_patch_size[1])
        x2 = x + src_patch_size[0]
        y2 = y + src_patch_size[1]

        # Save bounding box
        patch_coords.append((x, y, x2, y2))
        patches.append(patch)
        # Adjust the patch filename to match the previous code's format
        # print("patch sahpe ",patch.shape)
        # print("x_y ",x_y)
        # cv2.imshow("Image", patch)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    existing_patches += number_of_patches_per_page
    return patches,patch_coords


def draw_patches_on_image(page_img_path, patch_coords, main_root, color=(0, 0, 255), thickness=2):
    """
    Given the page image path and a list of patch bounding-box coordinates,
    draw rectangles on a color copy of the original image for visualization,
    then save that result to 'vis_windows' within the main_root.

    :param page_img_path: string, path to the original page image
    :param patch_coords: list of tuples (x1, y1, x2, y2) for each patch
    :param main_root: root directory where 'vis_windows' folder will be created
    :param color: (B, G, R) color for the rectangle border
    :param thickness: thickness of the rectangle border
    :return: The saved image path, or None if reading the image failed
    """

    # 1) Read the original image in color
    page_img = cv2.imread(page_img_path, cv2.IMREAD_COLOR)
    if page_img is None:
        print("Error reading image:", page_img_path)
        return None

    # 2) Draw each patch bounding box on the image
    for (x1, y1, x2, y2) in patch_coords:
        # Note: x1,y1 is (row, col), but for cv2.rectangle we use (col, row)
        cv2.rectangle(page_img, (y1, x1), (y2, x2), color, thickness)

    # 3) Build the output path inside "vis_windows" directory in main_root
    vis_dir = os.path.join(main_root, "vis_windows")
    os.makedirs(vis_dir, exist_ok=True)  # Make sure the folder exists

    # 4) Name the output file based on the original filename
    #    e.g., if page_img_path = "/path/to/file.jpg", we'll save "file.jpg" in vis_windows
    base_filename = os.path.basename(page_img_path)
    output_path = os.path.join(vis_dir, base_filename)

    # 5) Save the annotated image (no on-screen visualization)
    cv2.imwrite(output_path, page_img)
    print(f"Visualization saved to: {output_path}")

    return output_path




def denormalize_decade(value):
    """
    Denormalize the decades from the normalized range [0, 1] back to the original range.
    """
    return value * (MAX_DECADE - MIN_DECADE) + MIN_DECADE


def denormalize_year(normalized_year):
    """
    Denormalize the year back to its original range.
    """
    return int(normalized_year * (MAX_YEAR - MIN_YEAR) + MIN_YEAR)

def get_decade(year):
    """
    Convert year to decade.
    """
    return (year - MIN_YEAR) // 10

def predict_patch(image_path, model, device, exper_type="type-1"):
    """
    Predict the script type and year/decade from an input image.

    Args:
        image_path (str): Path to the input image.
        model (torch.nn.Module): Trained multitask model.
        device (str): Device to run inference on ("cuda" or "cpu").
        exper_type (str): Experiment type ("type-1", "type-2", "type-2-2").

    Returns:
        dict: Predicted script type, decade, and optional regression year.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    window_image = Image.fromarray(image_path).convert('RGB')
    window_tensor = transform(window_image).unsqueeze(0).to(device)
    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        if exper_type == "type-1":
            cls_logits, reg_output, _ = model(window_tensor)
            _, cls_idx = torch.max(cls_logits, dim=1)
            script_type = CATEGORY_MAPPING[cls_idx.item()]
            normalized_year = reg_output.squeeze().item()
            predicted_year = denormalize_year(normalized_year)
            predicted_decade = denormalize_decade(normalized_year)
            # print("script_type --> " , script_type)
            # print("predicted_year --> " , predicted_year)
            # print("predicted_decade --> " , predicted_decade)
            return script_type, predicted_year,predicted_decade
        else:
            raise ValueError(f"Invalid experiment type: {exper_type}")


def extract_image_info(image_path):
    """
    Extracts type and year information from a given image file name.

    Args:
        image_path (str): Path to the image file.

    Returns:
        dict: Dictionary containing Main-Type, Sub-Type, and Year.
    """
    # Extract the filename from the image path
    filename = os.path.basename(image_path)

    # Split the filename into its components
    # Expected format: [Main-Type]_[Sub-Type]_[Year]_[OtherInfo].jpg
    parts = filename.split('_')

    # Extract information based on the expected format
    if len(parts) < 4:
        raise ValueError("Filename does not conform to the expected format: Main-Type_Sub-Type_Year_OtherInfo.jpg")

    main_type = parts[0]
    sub_type = parts[1]
    year = parts[2].split('-')[0]  # This assumes the year is before a possible dash ('-')

    return main_type,sub_type,year


import cv2
from PIL import Image
import torchvision.transforms as transforms



#
# def extract_image_info(image_path):
#     """
#     Extracts type and year information from a given image file name.
#
#     Args:
#         image_path (str): Path to the image file.
#
#     Returns:
#         dict: Dictionary containing Main-Type, Sub-Type, and Year.
#     """
#     # Extract the filename from the image path
#     filename = os.path.basename(image_path)
#
#     # Split the filename into its components
#     # Expected format: [Main-Type]_[Sub-Type]_[Year]_[OtherInfo].jpg
#     parts = filename.split('_')
#
#     # Extract information based on the expected format
#     if len(parts) < 4:
#         raise ValueError("Filename does not conform to the expected format: Main-Type_Sub-Type_Year_OtherInfo.jpg")
#
#     main_type = parts[0]
#     sub_type = parts[1]
#     year = parts[2].split('-')[0]  # This assumes the year is before a possible dash ('-')
#
#     return {
#         'Main-Type': main_type,
#         'Sub-Type': sub_type,
#         'Year': year
#     }

def get_majority_and_median(types_list, years_list,decade_list):
    """
    Returns:
      - The majority class (most frequent item) in types_list.
      - The median value of years_list.
    """

    # 1) Find the majority class using a Counter
    label_counts = Counter(types_list)
    # print(label_counts)
    majority_class = label_counts.most_common(1)[0][0]  # e.g. "Italian_Square"

    # 2) Calculate the median - decade
    sorted_years = sorted(years_list)
    n = len(sorted_years)
    mid = n // 2

    if n % 2 == 1:
        # Odd number of elements -> single middle
        median_value = sorted_years[mid]
    else:
        # Even number of elements -> average of the two middle values
        median_value = (sorted_years[mid - 1] + sorted_years[mid]) / 2

    # 3) Calculate the median - year
    sorted_decades = sorted(decade_list)
    n_decade = len(decade_list)
    mid_deacde = n_decade // 2

    if n_decade % 2 == 1:
        # Odd number of elements -> single middle
        median_value_decade = sorted_decades[mid_deacde]
    else:
        # Even number of elements -> average of the two middle values
        median_value_decade = (sorted_decades[mid_deacde - 1] + sorted_decades[mid_deacde]) / 2

    return majority_class, median_value,median_value_decade


from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, confusion_matrix, accuracy_score, f1_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_performance_metrics(results):
    # Initialize lists to store ground truth and predictions
    gt_types, pred_types = [], []
    gt_main_types, pred_main_types = [], []
    gt_sub_types, pred_sub_types = [], []
    gt_years, pred_years = [], []
    gt_decades, pred_decades = [], []

    # Populate lists with data
    for result in results:
        gt_types.append(result['GT Type'])
        pred_types.append(result['Predicted Type'])
        gt_main_types.append(result['GT Main-Type'])
        pred_main_types.append(result['Predicted Main-Type'])
        gt_sub_types.append(result['GT Sub-Type'])
        pred_sub_types.append(result['Predicted Sub-Type'])
        gt_years.append(result['GT Year'])
        pred_years.append(result['Predicted Year'])
        gt_decades.append(result['GT Decade'])
        pred_decades.append(result['Predicted Decade'])

    # Calculate accuracy and other metrics for types
    accuracy_types = accuracy_score(gt_types, pred_types) * 100
    f1_types = f1_score(gt_types, pred_types, average='macro') * 100
    recall_types = recall_score(gt_types, pred_types, average='macro') * 100

    # Compute confusion matrices
    cm_types = confusion_matrix(gt_types, pred_types)
    cm_main_types = confusion_matrix(gt_main_types, pred_main_types)

    # Calculate MAE for years and decades
    mae_years = mean_absolute_error(gt_years, pred_years)
    mae_decades = mean_absolute_error(gt_decades, pred_decades)

    # Calculate L1 distances and their statistics
    l1_years = [abs(g - p) for g, p in zip(gt_years, pred_years)]
    l1_decades = [abs(g - p) for g, p in zip(gt_decades, pred_decades)]
    median_l1_years = np.median(l1_years)
    mean_l1_years = np.mean(l1_years)
    std_dev_l1_years = np.std(l1_years)

    median_l1_decades = np.median(l1_decades)
    mean_l1_decades = np.mean(l1_decades)
    std_dev_l1_decades = np.std(l1_decades)

    # Plot confusion matrices
    def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = 'd'  # 'd' means decimal integer
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):  # Use np.ndindex for indexing ndarray
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

    plot_confusion_matrix(cm_types, np.unique(gt_types), "Confusion Matrix for Types")
    plt.savefig('results/Swin-base/confusion_100_matrix_types.png')
    plt.show()

    plot_confusion_matrix(cm_main_types, np.unique(gt_main_types), "Confusion Matrix for Main Types")
    plt.savefig('results/Swin-base/confusion_100_matrix_main_types.png')
    plt.show()

    # Plot L1 distances
    plt.figure()
    plt.plot(l1_years, label='Year L1 Distances')
    plt.plot(l1_decades, label='Decade L1 Distances')
    plt.legend()
    plt.title('L1 Distances Over Samples')
    plt.xlabel('Sample Index')
    plt.ylabel('L1 Distance')
    plt.savefig('results/Swin-base/l1_100_distances.png')
    plt.show()

    # Save metrics to a text file
    with open('results/Swin-base/performance_100_summary.txt', 'w') as f:
        f.write(f"Model Accuracy Type: {accuracy_types:.2f}%\n")
        f.write(f"F1-Score Type: {f1_types:.2f}%\n")
        f.write(f"Recall Type: {recall_types:.2f}%\n")
        f.write(f"MAE Year: {mae_years:.2f}\n")
        f.write(f"MAE Decade: {mae_decades:.2f}\n")
        f.write(f"Median L1 Year: {median_l1_years:.2f}\n")
        f.write(f"Mean L1 Year: {mean_l1_years:.2f}\n")
        f.write(f"Standard Deviation L1 Year: {std_dev_l1_years:.2f}\n")
        f.write(f"Median L1 Decade: {median_l1_decades:.2f}\n")
        f.write(f"Mean L1 Decade: {mean_l1_decades:.2f}\n")
        f.write(f"Standard Deviation L1 Decade: {std_dev_l1_decades:.2f}\n")

    return {
        'Model Accuracy Type (%)': accuracy_types,
        'Model Error on Year': mae_years,
        'Model Error on Decade': mae_decades
    }

# def calculate_performance_metrics(results):
#     """
#     Calculate the accuracy for types and MAE for years and decades.
#
#     Args:
#         results (list of dicts): List containing dictionaries with comparison data.
#
#     Returns:
#         dict: Dictionary containing accuracy for types and MAE for years and decades.
#     """
#     # Initialize lists to store ground truth and predictions for types, years, and decades
#     gt_types = []
#     pred_types = []
#     gt_years = []
#     pred_years = []
#     gt_decades = []
#     pred_decades = []
#
#     # Populate lists with data
#     for result in results:
#         if result['Predicted Main-Type']:  # Check if prediction exists
#             gt_types.append(result['GT Main-Type'])
#             pred_types.append(result['Predicted Main-Type'])
#             gt_years.append(result['GT Year'])
#             pred_years.append(result['Predicted Year'])
#             gt_decades.append(result['GT Decade'])
#             pred_decades.append(result['Predicted Decade'])
#
#     # Calculate accuracy for types
#     correct_type_predictions = sum(1 for i in range(len(gt_types)) if gt_types[i] == pred_types[i])
#     accuracy_types = correct_type_predictions / len(gt_types) * 100
#
#     # Calculate MAE for years and decades
#     mae_years = mean_absolute_error(gt_years, pred_years)
#     mae_decades = mean_absolute_error(gt_decades, pred_decades)
#
#     return {
#         'Model Accuracy Type (%)': accuracy_types,
#         'Model Error on Year': mae_years,
#         'Model Error on Decade': mae_decades
#     }


import cv2
from PIL import Image
import torchvision.transforms as transforms


def predict_patch_path(image_path, model, device):
    """
    Predict the script type and year/decade from an input image path.

    Args:
        image_path (str): Path to the input image.
        model (torch.nn.Module): Trained multitask model.
        device (str): Device to run inference on ("cuda" or "cpu").

    Returns:
        Tuple containing the indices for script type, decade, and year.
    """
    # Load the image using cv2 to get a numpy array
    image = cv2.imread(image_path)
    if image is None:
        return

    # Convert BGR (OpenCV default) to RGB (what PIL expects)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the numpy array to a PIL Image
    pil_image = Image.fromarray(image).convert('RGB')

    # Define your transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply transforms
    window_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        cls_logits, reg_output, _ = model(window_tensor)
        _, cls_idx = torch.max(cls_logits, dim=1)
        script_type = CATEGORY_MAPPING[cls_idx.item()]
        decade_idx = reg_output.squeeze().item()
        year_idx = decade_idx  # Adjust as per your actual model output requirements

    return script_type, decade_idx, year_idx

def save_results_to_excel(results, output_path):
    """
    Save evaluation results to an Excel file.
    """
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False)
    print(f"Saved results to {output_path}")

def process_image_directory(directory_path, models_configs, target_images_per_page=20, visualize=False,experType="type-1"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    exper_type = "type-1"
    patch_size = (224,224)
    for model_config, weights_path in models_configs.items():
        model = MultiTaskModel(model_config, num_labels=14, num_decades=65, device=device, experType="type-1")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)

        for image_file in tqdm(os.listdir(directory_path)):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(directory_path, image_file)
                output = generate_patches_for_style(image_path, target_images_per_page, patch_size=patch_size)
                # draw_patches_on_image(image_path, coords, "out", color=(255, 0, 0), thickness=2)
                if output is None:
                    print(f"Skipping {image_path}: No patches generated.")
                    continue  # Skip this image if no patches were generated

                patches, coords = output  # Now safe to unpack
                if len(patches) < target_images_per_page * 0.5:  # If less than 70% of patches are valid
                    print("could not generate from this image ", image_path, " We generated: ",len(patches), "  Out Of: ",target_images_per_page)
                    continue

                #print("image_path --> ",image_path)
                GT_types, GT_decades , GT_years = [], [] , []
                types, decades , years = [], [] , []
                for patch in patches:
                    script_type_idx, decade_idx,yead_idx  = predict_patch(patch, model, device)
                    types.append(script_type_idx)
                    decades.append(decade_idx)
                    years.append(yead_idx)

                # Calculate majority and median
                majority, median_decade,median_year = get_majority_and_median(types, years, decades)
                MainType_GT,SubType_GT, Year_GT = extract_image_info(image_path)
                Year_GT = int(Year_GT)
                GT_Type = f'{MainType_GT}_{SubType_GT}'

                # print("GT_Type --> ",GT_Type)
                #
                # print(f"  Main-Type: {info['Main-Type']}")
                # print(f"  Sub-Type: {info['Sub-Type']}")
                # print(f"  Year: {info['Year']}\n")

                # print("GT vs pred (Type) --> ",GT_Type,majority)
                # print("GT vs pred (Year) --> ",Year_GT,median_year)
                # print("GT vs pred (Decade) --> ",get_decade(Year_GT),median_decade)


                # if visualize:
                #     visualize_results(image, coords, os.path.join("results", model_config, "visualizations"), image_file)
                # results.append({
                #     'Image Path': image_path,
                #     'GT Type': GT_Type,
                #     'Predicted Type': majority,
                #     'GT Year': Year_GT,
                #     'Predicted Year': median_year,
                #     'GT Decade': get_decade(Year_GT),
                #     'Predicted Decade': median_decade
                # })
                results.append({
                    'Image Path': image_path,
                    'GT Type': GT_Type,
                    'Predicted Type': majority,
                    'GT Main-Type': MainType_GT,
                    'Predicted Main-Type': majority.split("_")[0],
                    'GT Sub-Type': SubType_GT,
                    'Predicted Sub-Type': majority.split("_")[-1],
                    'GT Year': Year_GT,
                    'Predicted Year': median_year,
                    'GT Decade': get_decade(Year_GT),
                    'Predicted Decade': median_decade
                })

                # print("------ results ---")
                # print(results)
        performance_metrics = calculate_performance_metrics(results)
        print("Model Accuracy Type: {:.2f}%".format(performance_metrics['Model Accuracy Type (%)']))
        print("Model Error on Year: {:.2f}".format(performance_metrics['Model Error on Year']))
        print("Model Error on Decade: {:.2f}".format(performance_metrics['Model Error on Decade']))
    # # Save results to Excel
    save_results_to_excel(results,'results/NewExp-Type1/BeiT-base/summary_100_results.xlsx')



def extract_image_info(image_path):
    # Assuming the filename format is MainType_SubType_Year_ID.jpg
    parts = os.path.basename(image_path).split('_')
    main_type = parts[0]
    sub_type = parts[1]
    year = int(parts[2].split('_')[0])  # Year is before the first underscore in the third part
    return main_type, sub_type, year




def find_patches_for_page(data, main_type, sub_type, year, page_id):
    """
    Extract and print the paths of patches for a given page based on its metadata.

    Args:
        json_file (str): Path to the JSON file containing the dataset information.
        main_type (str): The main type of the manuscript (e.g., 'Ashkenazi').
        sub_type (str): The sub-type of the manuscript (e.g., 'Square').
        year (int): The year of the manuscript.
        page_id (str): The unique ID of the page (e.g., '158-0').
    """


    # Build the key from the given parameters
    key = f"{main_type}_{sub_type}"

    # Access the training data
    train_data = data['blind_test']
    patches = []
    # Check if the key exists in the training data
    if key in train_data:
        # Initialize a list to store the paths that match the given page ID and year
        matching_patches = []

        # Iterate over all patch paths under the key
        for path in train_data[key]:
            # Check if the path includes the page ID and year
            if page_id in path and str(year) in path:
                matching_patches.append(path)

        # Print the matching patch paths
        if matching_patches:
            return matching_patches
        else:
            print(f"No patches found for page ID {page_id} of {main_type} {sub_type} year {year}.")
    else:
        print(f"No data found for {key} in the training dataset.")

def load_json(file_path):
    """
    Load a JSON file from a specified filepath into a Python dictionary.

    Args:
    file_path (str): The path to the JSON file to be loaded.

    Returns:
    dict: A dictionary containing the data loaded from the JSON file.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            print("JSON file loaded successfully.")
            return data
    except FileNotFoundError:
        print("Error: The file was not found.")
        return None
    except json.JSONDecodeError:
        print("Error: The file is not a valid JSON.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Utilize the above function
if __name__ == "__main__":
    #     weights_path = r"/cs_storage/atamni/ServerBGU/Hebrew Paleography/Output/type-1/microsoft_beit-large-patch16-224/best_weights.pth"
    #     model_config = 'microsoft/beit-large-patch16-224'
    directory_path = 'images/blind_test_100'
    models_configs = {
      # 'google/vit-base-patch16-224': r"/cs_storage/atamni/ServerBGU/Hebrew Paleography/Output/type-1/google_vit-base-patch16-224/best_weights.pth"
       # 'microsoft/beit-large-patch16-224': r"/cs_storage/atamni/ServerBGU/Hebrew Paleography/Output/type-1/microsoft_beit-large-patch16-224/best_weights.pth"
       #'microsoft/beit-base-patch16-224': r"/cs_storage/atamni/ServerBGU/Hebrew Paleography/Output/type-1/microsoft_beit-base-patch16-224/best_weights.pth"
      'microsoft/beit-base-patch16-224': r"/cs_storage/atamni/ServerBGU/Hebrew Paleography/OutModelsExp/type-1/microsoft_beit-base-patch16-224/best_weights.pth"
    }
    file_path = 'JSON Dataset/4000_per_class/VML_Dataset_Paleography_4000.json'
    data = load_json(file_path)
    process_image_directory(directory_path, models_configs, visualize=True)

#
#
# def process_image_directory(json_data, directory_path, models_configs, visualize=False):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     results = []
#     for model_config, weights_path in models_configs.items():
#         model = MultiTaskModel(model_config, num_labels=len(CATEGORY_MAPPING), num_decades=65, device=device,experType="type-1")
#         model.load_state_dict(torch.load(weights_path, map_location=device))
#         model.to(device)
#
#         for image_file in tqdm(os.listdir(directory_path)):
#             if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 image_path = os.path.join(directory_path, image_file)
#                 main_type, sub_type, year = extract_image_info(image_path)
#                 image_id = image_path.split('_')[-1].split('.')[0]  # Extract image ID from filename
#
#                 patch_paths = find_patches_for_page(json_data, main_type, sub_type, year, image_id)
#                 print("image_path --> ",image_path, "found ",len(patch_paths))
#                 if not patch_paths:
#                     print("not patches for -->",image_path)
#                     continue  # Skip if no patches found
#
#                 types, years, decades = [], [], []
#                 for patch_path in patch_paths:
#                     if predict_patch_path(patch_path, model, device):
#                         script_type_idx, decade_idx, year_idx = predict_patch_path(patch_path, model, device)
#                         types.append(script_type_idx)
#                         years.append(year_idx)
#                         decades.append(decade_idx)
#
#                 if len(types) < 20:  # Check for minimum number of patches
#                     continue
#
#                 # Calculate majority and median
#                 majority_class = Counter(types).most_common(1)[0][0]
#                 median_year = sorted(years)[len(years) // 2]
#                 median_decade = sorted(decades)[len(decades) // 2]
#
#
#                 print("majority --> ",majority_class)
#                 print("median_year --> ",median_year)
#                 print("median_decade --> ",median_decade)
#
#                 results.append({
#                     'Image Path': image_path,
#                     'GT Main-Type': main_type,
#                     'Predicted Main-Type': majority_class.split("_")[0],
#                     'GT Sub-Type': sub_type,
#                     'Predicted Sub-Type': majority_class.split("_")[1],
#                     'GT Year': year,
#                     'Predicted Year': median_year,
#                     'GT Decade': get_decade(year),
#                     'Predicted Decade': median_decade
#                 })
#
#                 if visualize:
#                     # Implement visualization logic here if required
#                     pass
#
#         # Print or store performance metrics here
#         print_results(results)  # You need to implement this function to display or store results appropriately


# def main():
#     # Path to the image to predict
#     image_path = "images/Byzantine_Semi-cursive_1302_3325-0.jpg"
#     target_images_per_page = 100
#     patch_size = (224,224)
#     # Model and device configuration
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     weights_path = r"/cs_storage/atamni/ServerBGU/Hebrew Paleography/Output/type-1/microsoft_beit-large-patch16-224/best_weights.pth"
#     model_config = 'microsoft/beit-large-patch16-224'
#     num_labels = len(CATEGORY_MAPPING)
#     exper_type = "type-1"  # Can be "type-1", "type-2", "type-2-2"
#     patches,coords = generate_patches_for_style(image_path, target_images_per_page, patch_size=patch_size)
#
#     # Load the model
#     model = MultiTaskModel(model_config, num_labels=num_labels, num_decades=65, device=device, experType=exper_type)
#
#     try:
#         model.load_state_dict(torch.load(weights_path, map_location=device))
#         model.to(device)
#     except FileNotFoundError:
#         print(f"Skipping {model_config} due to missing weights.")
#         return
#
#     draw_patches_on_image(image_path, coords, "out", color=(255, 0, 0), thickness=2)
#     # Predict?
#     years = []
#     types = []
#     decades = []
#     for patche in patches:
#         script_type,predicted_years,predicted_decade = predict_patch(patche, model, device, exper_type)
#         types.append(script_type)
#         years.append(predicted_years)
#         decades.append(predicted_decade)
#
#     print("done")
#     print(types)
#     print(years)
#     majority, median_year ,median_decade  = get_majority_and_median(types, years,decades)
#
#     print("Majority Class :", majority)
#     print("Median Value Year :", median_year)
#     print("Median Value  Decade :", median_decade)
#
#
# if __name__ == "__main__":
#     main()
