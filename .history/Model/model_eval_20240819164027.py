import dropbox
import io
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
import json
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.metrics import precision_recall_curve
import huggingface_hub
from typing import List, Tuple, Dict
from accelerate import Accelerator
from torchvision.models.detection import maskrcnn_resnet50_fpn
from huggingface_hub import upload_file
import matplotlib.patches as patches
from torchvision.ops import roi_align
import torchvision.transforms.functional as F
import random
from random import sample
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
import random
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_curve, auc
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.metrics import ConfusionMatrixDisplay
from dotenv import load_dotenv

load_dotenv()
ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
dbx = dropbox.Dropbox(ACCESS_TOKEN)

# List files in a Dropbox folder
def list_files_in_folder(folder_path):
    files = []
    result = dbx.files_list_folder(folder_path)
    while True:
        files.extend([entry.name for entry in result.entries if isinstance(entry, dropbox.files.FileMetadata)])
        if not result.has_more:
            break
        result = dbx.files_list_folder_continue(result.cursor)
    return files

# Load JSON annotation from Dropbox
def load_annotation(dropbox_path):
    _, res = dbx.files_download(dropbox_path)
    return json.load(io.BytesIO(res.content))

# Load image from Dropbox
def load_image(dropbox_path):
    _, res = dbx.files_download(dropbox_path)
    file_bytes = io.BytesIO(res.content)
    return skio.imread(file_bytes)

# Plot image with bounding boxes
def plot_image_with_boxes(image, boxes, title="Image with Bounding Boxes"):
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)
    for box in boxes:
        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Create patches using torchvision.transforms.functional
def create_patches(image, boxes, patch_size):
    patches = []
    img_height, img_width = image.shape[:2]

    for i in range(0, img_height, patch_size):
        for j in range(0, img_width, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            patch_boxes = []
            for box in boxes:
                if (box[0] >= j and box[2] <= j + patch_size and
                        box[1] >= i and box[3] <= i + patch_size):
                    adjusted_box = [
                        box[0] - j,
                        box[1] - i,
                        box[2] - j,
                        box[3] - i
                    ]
                    patch_boxes.append(adjusted_box)
            if patch_boxes:
                patches.append((patch, patch_boxes))

    return patches

class GranzymeBDataset(Dataset):
    def __init__(self, image_dir: str, labels_dir: str, transform, patch_size: int = 256, val=False):
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.patch_size = patch_size
        self.val = val
        self.dbx = dropbox.Dropbox(ACCESS_TOKEN)
        self.images, self.labels = self._load_images_and_labels()

    def _list_files_in_folder(self, folder_path: str) -> List[str]:
        files = []
        result = self.dbx.files_list_folder(folder_path)
        while True:
            files.extend([entry.name for entry in result.entries if isinstance(entry, dropbox.files.FileMetadata)])
            if not result.has_more:
                break
            result = dbx.files_list_folder_continue(result.cursor)
        return files

    def _load_images_and_labels(self):
        label_files = self._list_files_in_folder(self.labels_dir)
        if self.val:
            # Use the first 6 samples for validation
            len_f = len(label_files)
            # select random 10% of the data for validation
            #make random
            label_files = sample(label_files, int(len_f*0.8))
            #label_files = label_files[:5]
        else:
            # Use the rest of the samples for training
            #use 80% of the data for training
            len_f = len(label_files)

            label_files = sample(label_files, int(len_f*0.8))
            #label_files = label_files[6:]

        image_files = set(self._list_files_in_folder(self.image_dir))

        images = []
        labels = []

        for label_file in label_files:
            sample_name = label_file.replace('_labels.json', '')
            img_name = sample_name + '_Granzyme B_path_view.tif'
            if img_name in image_files:
                images.append(img_name)
                labels.append(label_file)

        return images, labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_name = self.images[idx]
        label_name = self.labels[idx]

        # Read image path from Dropbox
        img_path = f"{self.image_dir}/{img_name}"
        image = load_image(img_path)
        #print(image.shape)

        # Read label path from Dropbox
        label_path = f"{self.labels_dir}/{label_name}"
        label_data = load_annotation(label_path)

        boxes = [box["Bounding Box"] for box in label_data]  # Ensure boxes is a list of lists

        patches = create_patches(image, boxes, self.patch_size)

        image_patches = []
        target_patches = []

        for patch, patch_boxes in patches:
            patch_labels = np.array([1 for _ in patch_boxes])

            if len(patch_boxes) == 0:
                patch_boxes = torch.zeros((0, 4), dtype=torch.float32)
                patch_labels = torch.zeros((0,), dtype=torch.int64)
                masks = torch.zeros((0, patch.shape[0], patch.shape[1]), dtype=torch.uint8)
            else:
                patch_boxes = torch.as_tensor(patch_boxes, dtype=torch.float32)
                patch_labels = torch.as_tensor(patch_labels, dtype=torch.int64)
                masks = torch.zeros((len(patch_boxes), patch.shape[0], patch.shape[1]), dtype=torch.uint8)  # Dummy masks

            target = {"boxes": patch_boxes, "labels": patch_labels, "masks": masks}

            if self.transform:
                #augmented = self.transform(image=patch)
                augmented = self.transform(image=patch, bboxes=patch_boxes, labels=patch_labels)
                patch = augmented['image']
                target = augmented['labels']
                patch_boxes = augmented['bboxes']
                patch = patch.clone().detach().float() / 255.0  # Normalize the image to [0, 1]

                patch_boxes = torch.as_tensor(patch_boxes, dtype=torch.float32)
                patch_labels = torch.as_tensor(patch_labels, dtype=torch.int64)


            else:
                augmented = self.transform(self.transform(image=patch, bboxes=patch_boxes, labels=patch_labels))
                patch = augmented['image']
                target = augmented['labels']
                patch_boxes = augmented['bboxes']
                patch = patch.clone().detach().float() / 255.0  # Normalize the image to [0, 1]

                patch_boxes = torch.as_tensor(patch_boxes, dtype=torch.float32)
                patch_labels = torch.as_tensor(patch_labels, dtype=torch.int64)

                

            target = {"boxes": patch_boxes, "labels": patch_labels, "masks": masks}


            image_patches.append(patch)
            target_patches.append(target)


        return image_patches, target_patches

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),  # Horizontal flip with 50% probability
        A.VerticalFlip(p=0.2),    # Vertical flip with 20% probability
        A.RandomBrightnessContrast(p=0.2),  # Randomly change brightness and contrast with 20% probability
        A.Rotate(limit=15, p=0.2),  # Rotate the image with 15 degrees limit and 20% probability
        A.Blur(blur_limit=3, p=0.2),  # Apply blur with 20% probability
        A.GaussNoise(p=0.2),  # Add Gaussian noise with 20% probability
        A.RandomScale(scale_limit=0.2, p=0.2),  # Randomly scale the image with 20% probability
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))



def get_val_transforms():
    return A.Compose([
        #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def custom_collate_fn(batch):
    image_patches = []
    target_patches = []

    for images, targets in batch:
        image_patches.extend(images)
        target_patches.extend(targets)

    return image_patches, target_patches

def get_model(num_classes: int):
    #changed default weights to None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    model.roi_heads.mask_predictor = None
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Training loop
def train(model, data_loader, val_loader, optimizer, scheduler, device, num_epochs):
    print('Start training')
    model.train()
    train_losses = []
    val_losses = []
    metrics = {"precision": [], "recall": [], "iou": []}

    for epoch in range(num_epochs):
        epoch_loss = 0
        for image_patches, target_patches in data_loader:
            for image, target in zip(image_patches, target_patches):
                plt.imshow(image.permute(1, 2, 0))
                if target["boxes"].numel() == 0:
                    continue

                optimizer.zero_grad()
                loss_dict = model([image.to(device)], [target])
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()
                epoch_loss += losses.item()

        scheduler.step()
        train_losses.append(epoch_loss / len(data_loader))
        val_loss, epoch_metrics = validate(model, val_loader, device)
        val_losses.append(val_loss)

        metrics["precision"].extend(epoch_metrics["precision"])
        metrics["recall"].extend(epoch_metrics["recall"])
        metrics["iou"].extend(epoch_metrics["iou"])

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(data_loader)}, Val Loss: {val_loss}")

    plot_losses(train_losses, val_losses)
    plot_precision_recall(metrics["precision"], metrics["recall"])
    plot_iou_histogram(metrics["iou"])

    #with open("metrics_small_2_5_transf_11", "w") as f:
    #    json.dump(metrics, f)

# Validation function
def validate(model, data_loader, device):
    #model.eval()
    val_loss = 0
    metrics = {"precision": [], "recall": [], "iou": []}

    with torch.no_grad():
        for image_patches, target_patches in data_loader:
            for image, target in zip(image_patches, target_patches):
                model.train()
                if target["boxes"].numel() == 0:
                    continue

                loss_dict = model([image.to(device)], [target])
                #print(loss_dict)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

                model.eval()
                epoch_metrics = calculate_metrics(model, image, target, device)
                metrics["precision"].extend(epoch_metrics["precision"])
                metrics["recall"].extend(epoch_metrics["recall"])
                metrics["iou"].extend(epoch_metrics["iou"])

    model.train()
    return val_loss / len(data_loader), metrics

def calculate_metrics(model, image, target, device):
    metrics = {"precision": [], "recall": [], "iou": []}

    with torch.no_grad():
        prediction = model([image.to(device)])[0]

    pred_boxes = prediction["boxes"].cpu().numpy()
    pred_scores = prediction["scores"].cpu().numpy()
    true_boxes = target["boxes"].cpu().numpy()

    if len(pred_scores) == 0 or len(true_boxes) == 0:
        # If there are no predictions or no true boxes, return default values
        return metrics

    # Generate binary labels for precision-recall calculation
    y_true = np.zeros_like(pred_scores, dtype=int)
    y_true[pred_scores > 0.5] = 1  # Using a threshold of 0.5 to determine positive predictions

    # Precision-Recall
    if len(np.unique(y_true)) > 1:
        precision, recall, _ = precision_recall_curve(y_true, pred_scores)
        metrics["precision"].extend(precision)
        metrics["recall"].extend(recall)
    else:
        metrics["precision"].extend([0])
        metrics["recall"].extend([0])

    # IoU
    ious = []
    for pred_box in pred_boxes:
        for true_box in true_boxes:
            iou = calculate_iou(pred_box, true_box)
            ious.append(iou)
    metrics["iou"].extend(ious)

    return metrics


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = intersection / float(box1_area + box2_area - intersection)
    return iou

# Plot training and validation losses
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color = 'purple')
    plt.plot(val_losses, label='Validation Loss', color= 'black')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.show()

# Plot precision-recall curve
def plot_precision_recall(precision, recall):
    plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, marker='.', color ='purple')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

# Plot histogram of IoU
def plot_iou_histogram(ious):
    plt.figure(figsize=(10, 5))
    plt.hist(ious, bins=50, color='purple', alpha=0.7)
    plt.xlabel('IoU')
    plt.ylabel('Frequency')
    plt.title('Distribution of IoU')
    plt.show()

# Plot image with predicted boxes and actual boxes
def plot_images_pred_boxes(image, pred_boxes, pred_scores, true_boxes):
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)
    for box, score in zip(pred_boxes, pred_scores):
        rect = plt.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            fill=False, edgecolor='red', linewidth=2
        )
        ax.add_patch(rect)
        plt.text(box[0], box[1], f'{score:.2f}', color='black', fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))


    for box in true_boxes:
        rect = plt.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            fill=False, edgecolor='blue', linewidth=2
        )
        ax.add_patch(rect)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Run inference and visualize results
def run_inference_and_visualize(model, data_loader, device):
    model.eval()
    for image_patches, target_patches in data_loader:
        for image, target in zip(image_patches, target_patches):
            if target["boxes"].numel() == 0:
                continue

            with torch.no_grad():
                prediction = model([image.to(device)])[0]

            img = image.cpu().permute(1, 2, 0).numpy()
            pred_boxes = prediction["boxes"].cpu().numpy()
            pred_scores = prediction["scores"].cpu().numpy()
            true_boxes = target["boxes"].cpu().numpy()

            plot_images_pred_boxes(img, pred_boxes, pred_scores, true_boxes)




# Plot ROC curve with confidence interval
def plot_roc_curve_with_ci(fpr, tpr, auc_score, n_bootstraps=1000):
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, color='purple', label=f'ROC curve (area = {auc_score:.2f})')

    # Bootstrap the ROC curve for confidence interval
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        indices = np.random.randint(0, len(fpr), len(fpr))
        if len(np.unique(indices)) < 2:
            continue

        fpr_resampled = resample(fpr, replace=True, n_samples=len(indices), random_state=i)
        tpr_resampled = resample(tpr, replace=True, n_samples=len(indices), random_state=i)

        roc_auc_resampled = auc(fpr_resampled, tpr_resampled)
        bootstrapped_scores.append(roc_auc_resampled)

    # Calculate the confidence interval
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # 95% confidence interval
    ci_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    ci_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    
    plt.fill_between(fpr, tpr - 0.1, tpr + 0.1, color='grey', alpha=0.3, label=f'95% CI: [{ci_lower:.2f} - {ci_upper:.2f}]')
    plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with Confidence Interval')
    plt.legend(loc='lower right')
    plt.show()

# Additional plotting functions for confusion matrix and mAP
def plot_confusion_matrix(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Purples)
    plt.title('Confusion Matrix')
    plt.show()

def plot_precision_recall(precision, recall):
    plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, marker='.', color='purple')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

# Cross-validation function
def cross_validate(model, dataset, k_folds=5):
    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_subset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        print(f"Starting Fold {fold+1}/{k_folds}")
        train(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs=10)
        fold_results.append(validate(model, val_loader, device))

    print("Cross-validation results:")
    for fold, result in enumerate(fold_results):
        print(f"Fold {fold+1}: mAP = {result[1]}, Validation Loss = {result[0]}")

def model_evaluation(model, dataset, device):
    model.eval()
    mAP_metric = MeanAveragePrecision()
    all_true_boxes = []
    all_pred_boxes = []
    all_pred_scores = []
    all_true_labels = []
    all_pred_labels = []

    data_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)
    val_loss = 0

    with torch.no_grad():
        for image_patches, target_patches in data_loader:
            for image, target in zip(image_patches, target_patches):
                if target["boxes"].numel() == 0:
                    continue

                loss_dict = model([image.to(device)], [target])
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

                prediction = model([image.to(device)])[0]
                mAP_metric.update([prediction], [target])

                all_true_boxes.append(target["boxes"].cpu().numpy())
                all_pred_boxes.append(prediction["boxes"].cpu().numpy())
                all_pred_scores.append(prediction["scores"].cpu().numpy())
                all_true_labels.append(target["labels"].cpu().numpy())
                all_pred_labels.append(prediction["labels"].cpu().numpy())

    mAP = mAP_metric.compute()["map"]
    precision, recall, _ = precision_recall_curve(np.hstack(all_true_labels), np.hstack(all_pred_scores))
    cm = confusion_matrix(np.hstack(all_true_labels), np.hstack(all_pred_labels))
    model.train()

    fpr, tpr, _ = roc_curve(np.hstack(all_true_labels), np.hstack(all_pred_scores))
    roc_auc = auc(fpr, tpr)

    # Plot ROC with confidence interval
    plot_roc_curve_with_ci(fpr, tpr, roc_auc)

    # Plot confusion matrix
    plot_confusion_matrix(cm)

    # Plot precision-recall curve
    plot_precision_recall(precision, recall)

    return val_loss / len(data_loader), mAP


if __name__ == "__main__":
    granzyme_b_image_folder = '/UEC, CD8 and GranzymeB'
    #manual_box_label
    labels_folder = '/Lables/manual_box_label'
    #'/Lables/manual_box_label'
    #flexible_simpie_fused_labels'
    patch_size = 256

    #dataset = GranzymeBDataset(granzyme_b_image_folder, labels_folder, transform=get_train_transforms(), patch_size=patch_size)
    #get_train_transforms()
    #data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)

    dataset_val = GranzymeBDataset(granzyme_b_image_folder, labels_folder, transform=get_val_transforms(), patch_size=patch_size, val=True)
    data_val_loader = DataLoader(dataset_val, batch_size=2, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model(num_classes=2)  # 2 classes: background and Granzyme B
    model.to(device)

    load_path = 'data/saved_models/10e_trans_with_relax.pth'

    model.load_state_dict(torch.load(load_path))
    #model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))


    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005) #lr = 0.0005 for 10 epochs fine tuning
    #    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005) original for 20 epochs of pretraining
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Train the model
    #train(model, data_loader,data_val_loader, optimizer, scheduler, device, num_epochs=1)
    #fused
    #simpie_flexible_only_training.pth
    #lablemanual_and_then_simpie
    #save_path = "data/saved_models/20e_trans_with_relax.pth" 
    #simpie_then_manual.pth has 3 epochs in simpie and 10 in manual
    #torch.save(model.state_dict(), save_path)

    # Run inference and visualize results
    model_evaluation(model, data_val_loader, device)


