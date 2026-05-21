import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, f1_score, mean_absolute_error, mean_squared_error, accuracy_score
from HebrewPaleopraphyLoader import HebrewDataset
from MultiTaskModel import MultiTaskModel
from ModelsConfigsNew import modelsConfigDict
import math

# Loss functions
criterion_cls = nn.CrossEntropyLoss()
criterion_mae = nn.L1Loss()
criterion_rmse = lambda preds, targets: torch.sqrt(nn.MSELoss()(preds, targets))

# Project Name
MAIN_PROJECT_NAME = "train-Type2"

def train_multitask(
    model, device, train_loader, valid_loader, optimizer, scheduler, 
    num_epochs, model_save_path, log_path, experType, alpha=1.0, beta=1.0, gamma=1.0, 
    early_stopping_patience=10
):
    model.to(device)
    best_valid_loss = float("inf")
    patience_counter = 0
    training_stats = {"train_loss": [], "valid_loss": [], "valid_accuracy": []}

    # Log model structure to W&B
    wandb.watch(model, log="all", log_freq=10)

    with open(log_path, "w") as log_file:
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
                inputs = inputs.to(device)
                optimizer.zero_grad()

                if experType == "type-1":
                    labels, decades = targets
                    labels, decades = labels.to(device), decades.to(device, dtype=torch.float32)
                    cls_logits, reg_decade, _ = model(inputs)
                    loss_cls = criterion_cls(cls_logits, labels)
                    loss_decade = criterion_mae(reg_decade.squeeze(), decades)
                    loss = alpha * loss_cls + beta * loss_decade

                # elif experType == "type-2":
                #    labels, decades = targets
                #    labels, decades = labels.to(device), decades.to(device)
                #    cls_logits, cls_decade, _ = model(inputs)
                #    loss_cls = criterion_cls(cls_logits, labels)
                #    loss_decade = criterion_cls(cls_decade, decades)
                #    loss = alpha * loss_cls + beta * loss_decade
                    
                elif experType == "type-2":
                    labels, decades = targets
                    labels = labels.to(device)
                    decades = decades.to(device).float()  # Ensure decades is a float tensor
                    cls_logits, cls_decade, _ = model(inputs)
                    loss_cls = criterion_cls(cls_logits, labels)

                    # If your cls_decade are logits, convert them to probabilities or actual values if necessary
                    # For instance, if it's a regression type of output:
                    # pred_decade = cls_decade.squeeze()
                    # If it's classification and you need class labels:
                    pred_decade = torch.argmax(cls_decade, dim=1).float()

                    # Calculate losses
                    loss_decade = criterion_mae(pred_decade, decades)  # Both should be float tensors
                    loss = alpha * loss_cls + beta * loss_decade

                else:  # "type-3"
                    labels,decades,years = targets
                    labels,decades, years= (labels.to(device),
                        decades.to(device, dtype=torch.float32),
                        years.to(device, dtype=torch.float32))
                    cls_logits, reg_decade, reg_year = model(inputs)
                    loss_cls = criterion_cls(cls_logits, labels)
                    loss_decade = criterion_mae(reg_decade.squeeze(), decades)
                    loss_year = criterion_rmse(reg_year.squeeze(), years)
                    loss = alpha * loss_cls + beta * loss_decade + gamma * loss_year

                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)
            training_stats["train_loss"].append(train_loss)

            # Validation
            model.eval()
            valid_loss = 0.0
            all_labels, all_preds = [], []
            mae, mse, mape, rmse = 0.0, 0.0, 0.0, 0.0
            for inputs, targets in tqdm(valid_loader, desc="Validation"):
                inputs = inputs.to(device)
                with torch.no_grad():
                    if experType == "type-1":
                        labels, decades = targets
                        labels, decades = labels.to(device), decades.to(device, dtype=torch.float32)
                        cls_logits, reg_decade, _ = model(inputs)
                        loss_cls = criterion_cls(cls_logits, labels)
                        loss_decade = criterion_mae(reg_decade.squeeze(), decades)
                        loss = alpha * loss_cls + beta * loss_decade
                        mae += mean_absolute_error(decades.cpu().numpy(), reg_decade.cpu().numpy())
                        mse += mean_squared_error(decades.cpu().numpy(), reg_decade.cpu().numpy())
                        mape += torch.mean(torch.abs((decades - reg_decade.squeeze()) / decades) * 100).item()
                        rmse += math.sqrt(mean_squared_error(decades.cpu().numpy(), reg_decade.cpu().numpy()))

                    elif experType == "type-2":
                        labels, decades = targets
                        labels = labels.to(device)
                        decades = decades.to(device).float()  # Ensure decades is a float tensor
                        cls_logits, cls_decade, _ = model(inputs)
                        loss_cls = criterion_cls(cls_logits, labels)

                        # If your cls_decade are logits, convert them to probabilities or actual values if necessary
                        # For instance, if it's a regression type of output:
                        # pred_decade = cls_decade.squeeze()
                        # If it's classification and you need class labels:
                        pred_decade = torch.argmax(cls_decade, dim=1).float()

                        # Calculate losses
                        loss_decade = criterion_mae(pred_decade, decades)  # Both should be float tensors
                        loss = alpha * loss_cls + beta * loss_decade

                        # Calculate metrics, assuming cls_decade is the predicted value and should be compared directly
                        mae += torch.nn.functional.l1_loss(pred_decade, decades, reduction='sum').item()
                        mse += torch.nn.functional.mse_loss(pred_decade, decades, reduction='sum').item()
                        rmse += torch.sqrt(torch.nn.functional.mse_loss(pred_decade, decades, reduction='sum')).item()
                        mape += torch.mean(torch.abs((decades - pred_decade) / decades) * 100).item()

                    else:  # "type-3"
                        labels,decades,years = targets
                        labels, decades, years = labels.to(device), decades.to(device, dtype=torch.float32),years.to(device, dtype=torch.float32)
                        cls_logits, reg_decade, reg_year = model(inputs)
                        loss_cls = criterion_cls(cls_logits, labels)
                        loss_decade = criterion_mae(reg_decade.squeeze(), decades)
                        loss_year = criterion_rmse(reg_year.squeeze(), years)
                        loss = alpha * loss_cls + beta * loss_decade + gamma * loss_year
                        mae += mean_absolute_error(decades.cpu().numpy(), reg_decade.cpu().numpy())
                        mse += mean_squared_error(decades.cpu().numpy(), reg_decade.cpu().numpy())
                        mape += torch.mean(torch.abs((decades - reg_decade.squeeze()) / decades) * 100).item()
                        rmse += math.sqrt(mean_squared_error(decades.cpu().numpy(), reg_decade.cpu().numpy()))

                    valid_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(cls_logits, 1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

            valid_loss /= len(valid_loader.dataset)
            valid_accuracy = sum([1 for i, j in zip(all_labels, all_preds) if i == j]) / len(all_labels)
            valid_recall = recall_score(all_labels, all_preds, average='macro')
            valid_f1 = f1_score(all_labels, all_preds, average='macro')
            mae /= len(valid_loader)
            mse /= len(valid_loader)
            mape /= len(valid_loader)
            rmse /= len(valid_loader)

            if experType == "type-1" or experType == "type-2" or experType == "all":
                # Update W&B
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "valid_accuracy": valid_accuracy,
                    "recall": valid_recall,
                    "f1": valid_f1,
                    "mae": mae,
                    "mse": mse,
                    "mape": mape,
                    "rmse": rmse
                })
            elif experType == "type-2":
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "valid_accuracy": valid_accuracy,
                    "recall": valid_recall,
                    "f1": valid_f1 ,
                    "mae": mae
                })


            # Log to text file
            log_file.write(
                f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
                f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.4f}, "
                f"Recall: {valid_recall:.4f}, F1: {valid_f1:.4f}, MAE: {mae:.4f}, "
                f"MSE: {mse:.4f}, MAPE: {mape:.4f}, RMSE: {rmse:.4f}\n"
            )

            scheduler.step(valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(model_save_path, "best_weights.pth"))
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break

    return training_stats


def main(exper_type, company_filter="apple,microsoft"):
    dataset_path = r"/cs_storage/atamni/ServerBGU/Hebrew Paleography/JSON Dataset/4000_per_class/VML_Dataset_Paleography_4000.json"
    full_dataset = HebrewDataset(
        json_path=dataset_path, split="train",augmentation_prob=0.7, visualization=False, label_flag=exper_type
    )
    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    selected_models = []
    if company_filter == "all":
        selected_models = [(company, config) for company, configs in modelsConfigDict.items() for config in configs]
    else:
        companies = company_filter.split(",")
        selected_models = [
            (company, config) for company in companies for config in modelsConfigDict.get(company, [])
        ]
    print("Selected Models ----> ",selected_models)
    for company, config in selected_models:
        wandb.init(project=MAIN_PROJECT_NAME, name=f"{config}_{exper_type}", group=exper_type)
        model = MultiTaskModel(config, num_labels=14, num_decades=65, device="cuda", experType=exper_type)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        model_save_path = f"OutModelsExp/{exper_type}/{config.replace('/', '_')}"
        #model_save_path = f"Output/type2-class-classMae/{config.replace('/', '_')}"
        os.makedirs(model_save_path, exist_ok=True)
        log_path = os.path.join(model_save_path, "training_log.txt")

        train_multitask(
            model, "cuda", train_loader, valid_loader, optimizer, scheduler, 30, model_save_path, log_path, exper_type
        )
        wandb.finish()


if __name__ == "__main__":
    main(exper_type="type-2", company_filter="all")
