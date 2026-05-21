import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
from MultiTaskModel import MultiTaskModel
from ModelsConfigsNew import modelsConfigDict

# Load category mapping
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

# Define transforms for the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def denormalize_year(normalized_year):
    """
    Denormalize the year back to its original range.
    """
    return int(normalized_year * (MAX_YEAR - MIN_YEAR) + MIN_YEAR)

def predict(image_path, model, device, exper_type="type-1"):
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
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Set the model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        if exper_type == "type-1":
            cls_logits, reg_output, _ = model(input_tensor)
            _, cls_idx = torch.max(cls_logits, dim=1)
            script_type = CATEGORY_MAPPING[cls_idx.item()]
            normalized_year = reg_output.squeeze().item()
            predicted_year = denormalize_year(normalized_year)
            
            return {
                "script_type": script_type,
                "predicted_year": predicted_year
            }

        elif exper_type == "type-2":
            cls_logits, cls_decade, _ = model(input_tensor)
            _, cls_idx = torch.max(cls_logits, dim=1)
            _, decade_idx = torch.max(cls_decade, dim=1)
            
            script_type = CATEGORY_MAPPING[cls_idx.item()]
            predicted_decade = decade_idx.item()
            
            return {
                "script_type": script_type,
                "predicted_decade": predicted_decade
            }
        
        elif exper_type == "type-2-2":
            cls_logits, cls_decade, _ = model(input_tensor)
            _, cls_idx = torch.max(cls_logits, dim=1)
            
            script_type = CATEGORY_MAPPING[cls_idx.item()]
            predicted_decade = denormalize_year(cls_decade.squeeze().item())
            
            return {
                "script_type": script_type,
                "predicted_decade": predicted_decade
            }
        
        else:
            raise ValueError(f"Invalid experiment type: {exper_type}")

def main():
    # Model and device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_path = r"/cs_storage/atamni/ServerBGU/Hebrew Paleography/Output/type-1/microsoft_beit-large-patch16-224/best_weights.pth"
    model_config = 'microsoft/beit-large-patch16-224'
    num_labels = len(CATEGORY_MAPPING)
    exper_type = "type-1"  # Can be "type-1", "type-2", "type-2-2"

    # Load the model
    model = MultiTaskModel(model_config, num_labels=14, num_decades=65, device=device, experType=exper_type)

    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
    except FileNotFoundError:
        print(f"Skipping {model_config} due to missing weights.")
        return

    # Path to the image to predict
    image_path = r"/cs_storage/atamni/ServerBGU/Hebrew Paleography/images/test/738-0.jpg"

    # Predict?
    predictions = predict(image_path, model, device, exper_type)
    print(f"Predictions: {predictions}")

if __name__ == "__main__":
    main()
