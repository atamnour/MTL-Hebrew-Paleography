import os
import torch
import pandas as pd
import wandb
import json
import matplotlib
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    recall_score, f1_score, accuracy_score, mean_absolute_error, mean_squared_error,mean_absolute_percentage_error, confusion_matrix, ConfusionMatrixDisplay
)
from HebrewPaleopraphyLoader import HebrewDataset
from MultiTaskModel import MultiTaskModel
from ModelsConfigsNew import modelsConfigDict

MAIN_PROJECT_NAME = "Evaluation-Type-2"

# Define normalization range (adjust to match your dataset)
MIN_DECADE = 0
MAX_DECADE = 65


def denormalize(value):
    """
    Denormalize the decades from the normalized range [0, 1] back to the original range,
    returning integer values.
    """
    denorm = value * (MAX_DECADE - MIN_DECADE) + MIN_DECADE
    return np.round(denorm).astype(int)


def evaluate_model(model, device, data_loader, test_type, exper_type):
    """
    Evaluate the model on the given dataset split.
    """
    model.eval()
    total_samples = 0
    metrics = {
        "Evaluate type": str(test_type),
        "cls_correct_type": 0,
        "cls_total": 0,
        "accuracy_type": 0.0,
        "MAE_decade": 0.0,
        "MSE_decade": 0.0,
        "RMSE_decade": 0.0,
    }
    # Lists to store all ground truth/predictions for decade regression
    all_gt_decades_denorm = []
    all_pred_decades_denorm = []

    # You can store all detailed results (both classification and regression) here
    results = []

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc=f"{test_type} Progress"):
            inputs = inputs.to(device)

            if exper_type == "type-1":
                img_path,labels, decades = targets
                labels, decades = labels.to(device), decades.to(device, dtype=torch.float32)
                cls_logits, reg_decade, _ = model(inputs)
                _, preds = torch.max(cls_logits, 1)
                # De-normalized metrics
                decades_denorm = denormalize(decades.cpu().numpy())
                reg_decade_denorm = denormalize(reg_decade.cpu().numpy())

                # print("preds_type(Model) --> ",preds)
                # print("labels_type(GT) --> ",labels)
                # print("preds_decade(Model) --> ",reg_decade)
                # print("labels_decade(GT) --> ",decades)
                # print("preds_decade_Real(Model) --> ", decades_denorm)
                # print("labels_decade_Unreal(GT) --> ", reg_decade_denorm)

                # Update metrics for classification and regression
                metrics["cls_correct_type"] += (preds == labels).sum().item()
                metrics["cls_total"] += labels.size(0)

                # Accumulate all GT/pred decades for final MAE calculation
                all_gt_decades_denorm.extend(decades_denorm)
                all_pred_decades_denorm.extend(reg_decade_denorm)


                # Store per-sample results for analysis
                # (If you actually have image paths, replace `str(inputs[i])` with the path.)
                for i in range(len(labels)):
                    anchor = "HebrewPatchesDataset"
                    parts = img_path[i].split(anchor, 1)  # Split on 'HebrewPatchesDataset' once
                    if len(parts) > 1:
                        new_path = anchor + parts[1]  # Reattach the anchor to the remainder
                    else:
                        new_path = path  # If 'HebrewPatchesDataset' not found, fallback
                    # print("new path --> ",new_path)
                    results.append({
                        "input_path": img_path,  # Or your actual path
                        "ground_truth_type": int(labels[i].cpu().item()),
                        "predicted_type": int(preds[i].cpu().item()),
                        "ground_truth_decade": int(decades_denorm[i]),
                        "predicted_decade": int(reg_decade_denorm[i]),
                    })

                # print("cls_correct_type(Model) --> ", metrics["cls_correct_type"])
                # print("cls_total(GT) --> ", metrics["cls_total"])
                # mae_norm = mean_absolute_error(decades_denorm, reg_decade_denorm)
                # mse_norm = mean_squared_error(decades_denorm, reg_decade_denorm)
                # mape_norm = mean_absolute_percentage_error(decades_denorm, reg_decade_denorm)
                # print("mae_norm --> ", mae_norm)
                # print("mse_norm --> ", mse_norm)
                # print("mape_norm --> ", mape_norm)


            elif exper_type == "type-2":
                img_path,labels, decades = targets
                labels, decades = labels.to(device), decades.to(device, dtype=torch.float32)
                decades = decades.long()

                cls_logits, cls_decade, _ = model(inputs)
                _, preds = torch.max(cls_logits, 1)
                # Predicted decades treated as regression output
                pred_decade = torch.argmax(cls_decade, dim=1)

                # _,pred_decade_max = torch.max(cls_decade, 1)
                # print("preds_type(Model) --> ",preds)
                # print("labels_type(GT) --> ",labels)
                # print("preds_decade_argmax(Model) --> ",pred_decade)
                # print("preds_type_max(Model) --> ",pred_decade_max)
                # print("labels_decade(GT) --> ",decades)

                # Update metrics
                metrics["cls_correct_type"] += (preds == labels).sum().item()
                metrics["cls_total"] += labels.size(0)

                # Accumulate all GT/pred decades for final MAE/MSE/RMSE calculation
                all_gt_decades_denorm.extend(decades.cpu().numpy())
                all_pred_decades_denorm.extend(pred_decade.cpu().numpy())

                # Store per-sample results for analysis
                # (If you actually have image paths, replace `str(inputs[i])` with the path.)
                for i in range(len(labels)):
                    anchor = "HebrewPatchesDataset"
                    parts = img_path[i].split(anchor, 1)  # Split on 'HebrewPatchesDataset' once
                    if len(parts) > 1:
                        new_path = anchor + parts[1]  # Reattach the anchor to the remainder
                    else:
                        new_path = path  # If 'HebrewPatchesDataset' not found, fallback
                    # print("new path --> ",new_path)
                    results.append({
                        "input_path": img_path,  # Or your actual path
                        "ground_truth_type": int(labels[i].cpu().item()),
                        "predicted_type": int(preds[i].cpu().item()),
                        "ground_truth_decade": int(decades[i].cpu().item()),
                        "predicted_decade": int(pred_decade[i].cpu().item())
                    })

                # mae_norm = mean_absolute_error(decades.cpu().numpy(), pred_decade.cpu().numpy())
                # mse_norm = mean_squared_error(decades.cpu().numpy(), pred_decade.cpu().numpy())
                # print("mae_norm --> ", mae_norm)
                # print("mse_norm --> ", mse_norm)

    if exper_type in ["type-1", "type-2"]:
        # ---- Compute final metrics (MAE,accuracy_type etc.) after the loop ----
        # For decade regression (denormalized)
        accuracy_type = metrics["cls_correct_type"] / metrics["cls_total"]
        mae_denorm_final = mean_absolute_error(all_gt_decades_denorm, all_pred_decades_denorm)
        mse_denorm_final = mean_squared_error(all_gt_decades_denorm, all_pred_decades_denorm)
        rmse_denorm_final = math.sqrt(mse_denorm_final)
        mape_denorm_final = mean_absolute_percentage_error(all_gt_decades_denorm, all_pred_decades_denorm)

        # Update aggregate metrics
        metrics["accuracy_type"] = round(accuracy_type, 2)
        metrics["MAE_decade"] = round(mae_denorm_final, 2)
        metrics["MSE_decade"] = round(mse_denorm_final, 2)
        metrics["RMSE_decade"] = round(rmse_denorm_final, 2)

    elif exper_type == "type-3":
        accuracy_type = metrics["cls_correct_type"] / metrics["cls_total"]
        mae_denorm_final = mean_absolute_error(all_gt_decades_denorm, all_pred_decades_denorm)
        mse_denorm_final = mean_squared_error(all_gt_decades_denorm, all_pred_decades_denorm)
        rmse_denorm_final = math.sqrt(mse_denorm_final)
        mape_denorm_final = mean_absolute_percentage_error(all_gt_decades_denorm, all_pred_decades_denorm)

        # Update aggregate metrics
        metrics["accuracy_type"] = round(accuracy_typeround, 2)
        metrics["MAE_decade"] = round(mae_denorm_final, 2)
        metrics["MSE_decade"] = round(mse_denorm_final, 2)
        metrics["RMSE_decade"] = round(rmse_denorm_final, 2)

    return metrics, results





# If running in an environment without a display (e.g., a remote server),
# you can uncomment the next line to use a non-interactive backend:
# matplotlib.use("Agg")
def visualize_samples(samples, save_path, title="Visualization"):
    """
    Create a 3x3 grid (up to 9 images) visualization from the given samples,
    then save it to disk without displaying on screen.

    Args:
        samples (list): A list of dicts, each with:
            - 'image':  A file path to an image on disk (str).
            - 'type':   The type label (str).
            - 'decade': The decade label (str).
        save_path (str): The path where the visualization (PNG) should be saved.
        title (str):     A title for the entire 3x3 figure.
    """
    # Only take at most 9 samples
    samples = samples[:9]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(title, fontsize=16)
    for i, sample in enumerate(samples):
        ax = axes[i // 3, i % 3]

        # (1) Get the single image path from the dictionary
        img_path = sample["image"]  # <-- FIXED HERE

        # (2) Load image from disk
        img_pil = Image.open(img_path)

        # (3) Display
        ax.imshow(img_pil)
        ax.axis("off")

        # (4) Set subplot title (type & decade labels)
        label_str = f"Type: {sample['type']}\nDecade: {sample['decade']}"
        ax.set_title(label_str, fontsize=12)

    # Hide any extra subplots if fewer than 9 samples
    total_samples = len(samples)
    for j in range(total_samples, 9):
        axes[j // 3, j % 3].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Leaves space for the main figure title

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def save_results_to_excel(results, output_path):
    """
    Save evaluation results to an Excel file.
    """
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False)
    print(f"Saved results to {output_path}")



def save_metrics_to_excel(metrics, output_dir, config, exper_type, split):
    """
    Save or append the 'metrics' dictionary as one row in an Excel file.
    The file will be saved in 'output_dir' under 'results_type.xlsx'.
    """
    # 1) Path to the Excel file (you can rename it if you prefer)
    excel_path = os.path.join(output_dir, "results_type_test.xlsx")

    # 2) Convert metrics to a DataFrame (with one row).
    #    Also add columns for config, exper_type, and split for clarity.
    df_metrics = pd.DataFrame([metrics])
    df_metrics["config"] = config
    df_metrics["exper_type"] = exper_type
    df_metrics["split"] = split

    # 3) If the file doesn't exist, create it. Otherwise, append.
    if not os.path.exists(excel_path):
        df_metrics.to_excel(excel_path, index=False)
    else:
        existing_df = pd.read_excel(excel_path)
        new_df = pd.concat([existing_df, df_metrics], ignore_index=True)
        new_df.to_excel(excel_path, index=False)


def save_all_metrics_to_excel(all_metrics, excel_path):
    """
    Write or append all model metrics (a list of dicts) to one Excel file.
    """
    df = pd.DataFrame(all_metrics)

    desired_cols = ["company", "config"] + [col for col in df.columns if col not in ["company", "config"]]
    df = df[desired_cols]

    # If you just want to overwrite/create this file each time:
    df.to_excel(excel_path, index=False)
    # If you want to create/overwrite:
    df.to_excel(excel_path, index=False)

    # Or if you want to append (check if file exists):
    # import os
    # if not os.path.exists(excel_path):
    #     df.to_excel(excel_path, index=False)
    # else:
    #     existing_df = pd.read_excel(excel_path)
    #     new_df = pd.concat([existing_df, df], ignore_index=True)
    #     new_df.to_excel(excel_path, index=False)


def evaluate_and_log_model(config, exper_type, weights_path, json_path, output_dir, batch_size=64, device="cuda"):
    """
    Evaluate the model and log results.
    """
    wandb.init(project=MAIN_PROJECT_NAME, name=f"{config}_{exper_type}", group=exper_type)

    model = MultiTaskModel(config, num_labels=14, num_decades=65, device=device, experType=exper_type)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except FileNotFoundError:
        print(f"Skipping {config} due to missing weights.")
        wandb.finish()
        return
    model.to(device)
    results_summary = {"test": [], "blind_test": []}
    for split in ["test"]:
        dataset = HebrewDataset(json_path=json_path, split=split,num_samples_per_class=600, augmentation_prob=0.0, label_flag=exper_type)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        metrics, results = evaluate_model(model, device, data_loader, split, exper_type)
        results_summary[split].append(metrics)
        wandb.log(metrics)
        # Save results to Excel
        save_results_to_excel(results, os.path.join(output_dir, f"{split}_predictions.xlsx"))
        # Save aggregate metrics to "results_type.xlsx" in the root directory
        save_metrics_to_excel(metrics, output_dir, config, exper_type, split)
        # # 9 samples to visualize
        # vis_samples = []
        # print("aaaaaaaaaaaaaaaaaaaa")
        # for i in range(min(9, len(results))):
        #     # Create strings for the type and decade display:
        #     type_str = f'{results[i]["ground_truth_type"]} / {results[i]["predicted_type"]}'
        #     decade_str = f'{results[i]["ground_truth_decade"]} / {results[i]["predicted_decade"]}'
        #     path = results[i]["input_path"]
        #     print(path)
        #     print(path.type)
        #
        #     exit()
        #     # Append one sample dictionary (one image path + its labels)
        #     vis_samples.append({
        #         "image": results[i]["input_path"],  # Single path string
        #         "type": type_str,
        #         "decade": decade_str
        #     })
        #
        #
        # print("vis_samples ",vis_samples)
        # # Now call the visualization function
        # # str1 = f'{split}_visualization.png'
        # test_vis_path = os.path.join(output_dir, "visualization", "test_visualization.png")
        # os.makedirs(os.path.dirname(test_vis_path), exist_ok=True)
        #
        # visualize_samples(vis_samples, test_vis_path, title="Test Visualization (Sample)")
    wandb.finish()
    return results_summary




def main(exper_type, weights_root_dir, json_path, company_filter="all", device="cuda"):
    """
    Main evaluation function for all models and configurations.
    """
    selected_models = []
    results_summary = {}
    if company_filter == "all":
        selected_models = [(company, config) for company, configs in modelsConfigDict.items() for config in configs]
    else:
        companies = company_filter.split(",")
        selected_models = [
            (company, config) for company in companies for config in modelsConfigDict.get(company, [])
        ]

    # 2) This list will store metrics from every model+split
    all_metrics = []

    for company, config in selected_models:
        weights_path = os.path.join(weights_root_dir, exper_type, config.replace("/", "_"), "best_weights.pth")
        output_dir = os.path.join(weights_root_dir, exper_type, config.replace("/", "_"))

        if os.path.exists(weights_path):
            results_summary = evaluate_and_log_model(config, exper_type, weights_path, json_path, output_dir, device=device)
            # Collect metrics and store them in 'all_metrics'
            for split, metrics_list in results_summary.items():
                for metric_dict in metrics_list:
                    # Add identifying info (company, config, split)
                    metric_dict["company"] = company
                    metric_dict["config"] = config
                    # metric_dict["split"] = split
                    all_metrics.append(metric_dict)
        else:
            print(f"Weights not found for config '{config}', skipping.")

    # 4) After ALL models are processed, save the entire 'all_metrics' to a single Excel
    if all_metrics:  # If the list is not empty
        save_all_metrics_to_excel(all_metrics, os.path.join(weights_root_dir,exper_type, f"{exper_type}_results_all.xlsx"))
    else:
        print("No metrics to save (no valid models found).")


if __name__ == "__main__":
    weights_root_dir = "OutModelsExp/NewExp"
    json_path = r"/cs_storage/atamni/ServerBGU/Hebrew Paleography/JSON Dataset/4000_per_class/VML_Dataset_Paleography_4000.json"
    main(exper_type="type-1", weights_root_dir=weights_root_dir, json_path=json_path, company_filter="all")
