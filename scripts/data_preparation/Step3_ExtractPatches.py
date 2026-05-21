import _thread
import os
# from util import generate_patches
import shutil
import numpy as np
import os
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

########################################################################
########################################################################
########################################################################


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


def get_random_patch_location(img, patch_size=(224, 224)):  # 350,350
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


def generate_patches_for_style(style_path, output_class_folder, number_of_patches_per_page, patch_size,
                               number_of_lines_per_patch=5, margine_p_x=0.1, margine_p_y=0.1):
    existing_patches = 0  # Start patch count
    for page in tqdm(os.listdir(style_path)):
        page_path = os.path.join(style_path, page)
        page_img = cv2.imread(page_path, 0)

        if (min(page_img.shape) < patch_size[0]):
            print('Error page too small', page_path)
            continue

        init_patch_size = (page_img.shape[0] // 10, page_img.shape[0] // 10)

        avg_lines_num = calculate_average_number_of_line_in_patch(page_img, patch_size=init_patch_size)

        if avg_lines_num < 0:
            print('Error couldn\'t calculate avg number of line in page', page_path)
            continue

        src_patch_size = (int(init_patch_size[0] * (number_of_lines_per_patch / avg_lines_num)),
                          int(init_patch_size[1] * (number_of_lines_per_patch / avg_lines_num)))

        for i in tqdm(range(number_of_patches_per_page), desc=f'Generating patches for {page}'):
            patch, x_y = get_valid_patch(page_img, src_patch_size=src_patch_size, dst_patch_size=patch_size,
                                         margine_p_x=margine_p_x, margine_p_y=margine_p_y)

            if patch is None:
                print(f'Error: unable to generate patches from page {page_path}')
                break

            # Adjust the patch filename to match the previous code's format
            patch_filename = f"{os.path.splitext(page)[0]}_patchpach_{existing_patches + i + 1}.png"
            patch_path = os.path.join(output_class_folder, patch_filename)
            cv2.imwrite(patch_path, patch)

        existing_patches += number_of_patches_per_page

def generate_patches(data_folder, output_folder, patch_size, target_images_per_page=100):
    output_folder = f'{output_folder}_{patch_size[0]}x{patch_size[1]}'
    os.makedirs(output_folder, exist_ok=True)

    for style_folder in os.listdir(data_folder):
        style_path = os.path.join(data_folder, style_folder)
        output_class_folder = os.path.join(output_folder, style_folder)
        os.makedirs(output_class_folder, exist_ok=True)

        images = [file for file in os.listdir(style_path) if file.endswith(('jpg', 'jpeg', 'png', 'gif'))]
        num_images = len(images)
        print(images)
        if num_images >= target_images_per_page:
            print(f"Skipping '{style_folder}' as it already has {num_images} images")
            continue

        print(f"Generating patches for '{style_folder}'")

        generate_patches_for_style(style_path, output_class_folder, target_images_per_page, patch_size=patch_size)


data_folder = "../../data/dataset_pages_annotated/Sefardic/Square"  # Replace with your dataset path
output_folder = "../../patches/Sefardic-Square"

generate_patches(
    data_folder,
    output_folder,
    patch_size=(224, 224),
    target_images_per_page=100
)
