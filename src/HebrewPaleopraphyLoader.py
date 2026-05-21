import os
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

class HebrewDataset(Dataset):
    def __init__(self, json_path, split='train', num_samples_per_class=None, augmentation_prob=0.5,
                 visualization=False, visualization_samples=10, visualization_folder="loadervisualization",
                 min_year=895, max_year=1540, label_flag='all', save_decade_map=False, epsilon=1e-10):
        """
        Initialize the dataset from a JSON file with decade calculation and flexible label selection.
        Args:
            json_path (str): Path to the JSON file containing image paths.
            split (str): Dataset split to use ('train', 'valid', or 'test').
            num_samples_per_class (int, optional): Number of samples to load per class.
            augmentation_prob (float): Probability of applying augmentations.
            visualization (bool): If True, visualizes images before and after augmentation.
            visualization_samples (int): Number of visualization samples per class.
            visualization_folder (str): Directory to save visualization images.
            min_year (int): Minimum year for normalization (default: 895).
            max_year (int): Maximum year for normalization (default: 1540).
            label_flag (str): Label type to return ('all', 'year', 'type', 'decade').
            save_decade_map (bool): If True, saves the decade map to a JSON file.
            epsilon (float): Small value to add to normalization to prevent zeros.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.imagesPaths = data.get(split, {})

        # Debugging: Print all categories found in JSON
        print("Categories loaded from JSON:", self.imagesPaths.keys())

        # Ensure all files exist and are counted correctly
        self.imagesPaths = {
            label: [img_path for img_path in images if os.path.exists(img_path)]
            for label, images in self.imagesPaths.items()
        }

        # Debugging: Print categories after checking for file existence
        print("Categories after existence check:", self.imagesPaths.keys())

        # Remove empty classes if any
        self.imagesPaths = {label: images for label, images in self.imagesPaths.items() if images}
        print("Categories with non-empty images:", self.imagesPaths.keys())

        # Mapping categories to indices
        self.category_to_label = {name: idx for idx, name in enumerate(self.imagesPaths.keys())}
        print("Final category to label mapping:", self.category_to_label)
        # Sample a limited number of images per class
        if num_samples_per_class:
            self.imagesPaths = {label: images[:num_samples_per_class] for label, images in self.imagesPaths.items()}

        # Define transformations with conditional augmentation
        self.augmentation_prob = augmentation_prob
        self.visualization = visualization
        self.visualization_samples = visualization_samples
        self.visualization_folder = visualization_folder

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BILINEAR),
            transforms.RandomApply([
                transforms.RandomRotation(degrees=(-10, 10)),
                transforms.GaussianBlur(kernel_size=3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip()
            ], p=self.augmentation_prob),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Map category names to label indices
        self.category_names = list(self.imagesPaths.keys())
        self.category_to_label = {name: idx for idx, name in enumerate(self.category_names)}

        # Decade mapping
        self.min_year = min_year
        self.max_year = max_year
        self.decades = self._build_decade_map(min_year, max_year)

        # Save decade map if required
        if save_decade_map:
            with open("decade_map.json", "w") as f:
                json.dump(self.decades, f, indent=4)

        # Label flag
        self.label_flag = label_flag
        self.epsilon = epsilon  # Small value to prevent zeros in normalized values

        # Prepare the visualization folder
        if self.visualization:
            os.makedirs(self.visualization_folder, exist_ok=True)

    def _build_decade_map(self, min_year, max_year):
        """
        Build a map of decade indices to year ranges.
        """
        decade_map = {}
        decade_idx = 0
        for start_year in range(min_year, max_year + 1, 10):
            end_year = start_year + 9
            decade_map[decade_idx] = (start_year, end_year)
            decade_idx += 1
        return decade_map

    def _get_decade(self, year):
        """
        Get the decade index for a given year.
        """
        for idx, (start, end) in self.decades.items():
            if start <= year <= end:
                return idx
        raise ValueError(f"Year {year} is out of range {self.min_year}-{self.max_year}.")

    def __len__(self):
        # Count total images across all classes
        return sum(len(images) for images in self.imagesPaths.values())

    def __getitem__(self, idx):
        # Convert the linear index to a specific class/image path
        category, img_path = self.get_path_by_index(idx)

        # Extract the year from the file path
        year = self.extract_year(img_path)
        normalized_year = self.normalize_year(year)
        decade = self._get_decade(year)
        normalized_decade = self.normalize_decade(decade)

        # Load the image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img_pil = Image.fromarray(img)  # Convert to PIL Image

        # Apply transformations (with augmentation)
        img_aug = self.transform(img_pil)

        # Visualization and saving
        if self.visualization:
            self.save_visualization_image(category, img_pil, img_aug, img_path, year, decade)

        # Define the label (category) as the target
        label = self.category_to_label[category]
        # print("img_path ---> ",img_path)
        # print("decade ---> ",decade,"    Norm --> ",normalized_decade)
        # print("year ---> ",year,"    Norm --> ",normalized_year)

        # Return based on the label flag
        if self.label_flag == "type-1":
            return img_aug, (label, max(normalized_decade, self.epsilon))
        if self.label_flag == "type-2":
            return img_aug, (label, decade)
        if self.label_flag == "type-3":
            return img_aug, (label, max(normalized_year, self.epsilon))
        if self.label_flag == 'year':
            return img_aug, max(normalized_year, self.epsilon)
        elif self.label_flag == 'type':
            return img_aug, label
        elif self.label_flag == 'decade':
            return img_aug, max(decade, self.epsilon)
        else:  # 'all'
            return img_aug, (label, max(normalized_decade, self.epsilon), max(normalized_year, self.epsilon))

    def get_path_by_index(self, idx):
        """
        Convert a flat index to category and image path.
        """
        cumulative_count = 0
        for category, images in self.imagesPaths.items():
            if cumulative_count + len(images) > idx:
                img_path = images[idx - cumulative_count]
                return category, img_path
            cumulative_count += len(images)
        raise IndexError(f"Index {idx} out of range")

    def extract_year(self, img_path):
        """
        Extract the year from the file path.
        Assumes the year is included as a folder name in the file path.
        """
        parts = img_path.split(os.sep)
        for part in parts:
            if part.isdigit() and self.min_year <= int(part) <= self.max_year:
                return int(part)
        raise ValueError(f"Year not found in path: {img_path}")

    def normalize_year(self, year):
        """
        Normalize the year to a value between 0 and 1.
        """
        return (year - self.min_year) / (self.max_year - self.min_year)

    def normalize_decade(self, decade):
        """
        Normalize the decade to a value between 0 and 1.
        """
        min_decade = 0
        max_decade = (self.max_year - self.min_year) / 10
        return (decade - min_decade) / (max_decade - min_decade)

    def save_visualization_image(self, category, img_pil, img_aug, img_path, year, decade):
        """
        Saves a combined visualization of the original and augmented images with metadata.
        """
        category_folder = os.path.join(self.visualization_folder, category)
        os.makedirs(category_folder, exist_ok=True)

        # Combine original and augmented images
        img_pil_resized = img_pil.resize((224, 224))
        img_aug_resized = transforms.ToPILImage()(img_aug).resize((224, 224))
        combined_image = Image.new('RGB', (448, 224))
        combined_image.paste(img_pil_resized, (0, 0))
        combined_image.paste(img_aug_resized, (224, 0))

        # Add metadata
        draw = ImageDraw.Draw(combined_image)
        font = ImageFont.load_default()
        text = f"Path: {img_path}\nYear: {year}\nDecade: {decade} ({self.decades[decade][0]}-{self.decades[decade][1]})"
        draw.text((10, 10), text, fill="white", font=font)

        # Save the combined image
        save_path = os.path.join(category_folder, f"visualization_{len(os.listdir(category_folder)) + 1}.png")
        combined_image.save(save_path)


def verify_label_mapping(loader_mapping, predefined_mapping):
    """
    Check if the loader's category-to-label mapping matches the predefined category mapping.

    Args:
        loader_mapping (dict): Mapping from category names to labels from the dataset loader.
        predefined_mapping (dict): Predefined mapping of indices to category names.

    Returns:
        bool: True if the mappings are synchronized, False otherwise.
    """
    # Reverse the loader mapping for comparison
    loader_mapping_reversed = {v: k for k, v in loader_mapping.items()}

    # Check each item in the predefined mapping
    for idx, name in predefined_mapping.items():
        if loader_mapping_reversed.get(idx) != name:
            print(f"Mismatch found: Index {idx} should be '{name}', but got '{loader_mapping_reversed.get(idx)}'")
            return False

    print("All mappings are correctly synchronized.")
    return True
# # Load category mapping
# CATEGORY_MAPPING = {
#     0: 'Ashkenazi_Square', 1: 'Ashkenazi_Semi-cursive', 2: 'Ashkenazi_Cursive',
#     3: 'Byzantine_Square', 4: 'Byzantine_Semi-cursive', 5: 'Italian_Square',
#     6: 'Italian_Semi-cursive', 7: 'Oriental_Square', 8: 'Oriental_Semi-cursive',
#     9: 'Sefardic_Square', 10: 'Sefardic_Semi-cursive', 11: 'Sefardic_Cursive',
#     12: 'Yemenite_Square', 13: 'Yemenite_Semi-cursive'
# }
# # # Example Usage
# json_path = r"/cs_storage/atamni/ServerBGU/Hebrew Paleography/JSON Dataset/4000_per_class/VML_Dataset_Paleography_4000.json"
# dataset = HebrewDataset(
#      json_path=json_path,
#      split="train",
#      augmentation_prob=0.5,
#      visualization=True,
#      visualization_folder="visualizations",
#      min_year=895,
#      max_year=1540,
#      label_flag='all',
#      save_decade_map=True
#  )
# loader_mapping = dataset.category_to_label
# print(dataset.category_to_label)
# print(len(dataset.category_to_label))
# predefined_mapping = CATEGORY_MAPPING
#
# # Check if mappings are synchronized
# verify_label_mapping(loader_mapping, predefined_mapping)
# # # # Dataloader example
# # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
# # for img, labels in dataloader:
# #      print(f"Labels: {labels}")
# #      break
