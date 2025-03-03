import os
import io
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import dropbox
from skimage import io as skio
from dotenv import load_dotenv
from typing import List, Tuple, Dict

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.ops import roi_align

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.utils import resample

import matplotlib.patches as patches
from huggingface_hub import upload_file
from accelerate import Accelerator

"""
This script is used to evaluate the model using mean Average Precision (mAP) and IoU.
It loads the train model to use in a validation dataset. 

"""

#Loads the enviromental variable for Dropbox access
load_dotenv()
ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
dbx = dropbox.Dropbox(ACCESS_TOKEN)


def list_files_in_folder(folder_path):
    """
    Lists all files in a Dropbox folder.
    
    Args:
        folder_path (str): Path to the folder in Dropbox.

    Returns:
        list: List of file names in the folder.
    """
    files = []
    result = dbx.files_list_folder(folder_path)
    while True:
        files.extend([entry.name for entry in result.entries if isinstance(entry, dropbox.files.FileMetadata)])
        if not result.has_more:
            break
        result = dbx.files_list_folder_continue(result.cursor)
    return files

def load_annotation(dropbox_path):
    """
    Loads JSON annotation from Dropbox. The labels come from Labelme.
    
    Args:
        dropbox_path (str): Path to the JSON file in Dropbox.

    Returns:
        dict: Parsed JSON annotation.
    """
    _, res = dbx.files_download(dropbox_path)
    return json.load(io.BytesIO(res.content))

# Load image from Dropbox
def load_image(dropbox_path):
    """
    Loads an image from Dropbox.
    
    Args:
        dropbox_path (str): Path to the image file in Dropbox.

    Returns:
        np.ndarray: Image array.
    """
    _, res = dbx.files_download(dropbox_path)
    file_bytes = io.BytesIO(res.content)
    return skio.imread(file_bytes)

def plot_image_with_boxes(image, boxes, title="Image with Bounding Boxes"):
    """
    Plots an image with bounding boxes overlaid.
    
    Args:
        image (np.ndarray): The image to be displayed.
        boxes (List[List[int]]): A list of bounding boxes, each represented as [x1, y1, x2, y2].
        title (str): Title for the plot.
    """
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
    """
    Creates patches from an image along with adjusted bounding boxes.
    
    Args:
        image (np.ndarray): Input image.
        boxes (List[List[int]]): List of bounding boxes in the original image.
        patch_size (int): Size of each patch.

    Returns:
        List[Tuple[np.ndarray, List[List[int]]]]: A list of patches with their corresponding bounding boxes.
    """
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
        """
        Custom dataset for loading images and labels from Dropbox.
        
        Args:
            image_dir (str): Path to the image directory.
            labels_dir (str): Path to the labels directory.
            transform: Data augmentation transformations.
            patch_size (int): Size of patches to generate.
            val (bool): Whether this dataset is for validation.
        """
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.patch_size = patch_size
        self.val = val
        self.dbx = dropbox.Dropbox(ACCESS_TOKEN)
        self.images, self.labels = self._load_images_and_labels()

    def _list_files_in_folder(self, folder_path: str) -> List[str]:
        """Lists all files in a Dropbox folder."""
        files = []
        result = self.dbx.files_list_folder(folder_path)
        while True:
            files.extend([entry.name for entry in result.entries if isinstance(entry, dropbox.files.FileMetadata)])
            if not result.has_more:
                break
            result = dbx.files_list_folder_continue(result.cursor)
        return files

    def _load_images_and_labels(self):
        """Loads and matches images with corresponding label files."""
        label_files = self._list_files_in_folder(self.labels_dir)
        if self.val:
            ## CHANGE HERE: depending on the validation method desired 
            # Use the first 6 samples for validation (in the data set used). 
            len_f = len(label_files) 
            # select random 10% of the data for validation
            #make random
            #label_files = sample(label_files, int(len_f*0.8))
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
        """Loads an image and corresponding label, creates patches, and applies transformations."""
        img_name = self.images[idx]
        label_name = self.labels[idx]

        # Read image path from Dropbox
        img_path = f"{self.image_dir}/{img_name}"
        image = load_image(img_path)

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


def get_val_transforms():
    """
    Returns the transformation pipeline for validation data.
    Converts images to tensors without additional augmentations.
    """
    return A.Compose([
        #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader to handle batches of image patches.
    
    Args:
        batch (list): List of tuples containing image patches and target patches.
    
    Returns:
        tuple: Two lists containing image patches and target patches.
    """
    image_patches = []
    target_patches = []

    for images, targets in batch:
        image_patches.extend(images)
        target_patches.extend(targets)

    return image_patches, target_patches

def get_model(num_classes: int):
    """
    Loads a Mask R-CNN model with a modified head for object detection. It uses only the boudning box, not the mask. 
    
    Args:
        num_classes (int): Number of classes including background.

    Returns:
        torch.nn.Module: Modified Mask R-CNN model.
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    model.roi_heads.mask_predictor = None
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def calculate_metrics(model, image, target, device):
    """
    Computes evaluation metrics including precision, recall, and IoU.
    
    Args:
        model (torch.nn.Module): The trained object detection model.
        image (torch.Tensor): The input image tensor.
        target (dict): Ground truth bounding boxes and labels.
        device (torch.device): Device to run inference on (CPU/GPU).
    
    Returns:
        dict: Dictionary containing precision, recall, and IoU lists.
    """
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
    y_true[pred_scores > 0.5] = 1  

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
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1 (list): Coordinates [x1, y1, x2, y2] of the first bounding box.
        box2 (list): Coordinates [x1, y1, x2, y2] of the second bounding box.
    
    Returns:
        float: IoU score between the two boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = intersection / float(box1_area + box2_area - intersection)
    return iou


# Plot precision-recall curve
def plot_precision_recall(precision, recall):
    """
    Plots the Precision-Recall curve.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, marker='.', color ='purple')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

# Plot histogram of IoU
def plot_iou_histogram(ious):
        """
    Plots a histogram of Intersection over Union (IoU) values.
    """
    plt.figure(figsize=(10, 5))
    plt.hist(ious, bins=50, color='purple', alpha=0.7)
    plt.xlabel('IoU')
    plt.ylabel('Frequency')
    plt.ylim(0, 20)
    plt.title('Distribution of IoU')
    plt.show()

def plot_iou_histogram_50(ious):
    plt.figure(figsize=(10, 5))
    plt.hist(ious, bins=50, color='purple', alpha=0.7)
    plt.xlabel('IoU')
    plt.ylabel('Frequency')
    plt.ylim(0, 10)
    plt.title('Distribution of IoU >= 50')
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

def model_evaluation(model, data_loader, device):
    """
    Evaluates the model using mean Average Precision (mAP) and IoU metrics.
    
    Args:
        model (torch.nn.Module): The trained detection model.
        data_loader (DataLoader): DataLoader for validation/test data.
        device (torch.device): Device for computation (CPU/GPU).

    Returns:
        overall_result (dict): Computed evaluation metrics.
        patch_metrics (dict): 
    """
    model.eval()
    overall_metric = MeanAveragePrecision(class_metrics=True)
    recall_50 = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)
    #iou_thresholds=[0.5], 
    patch_metrics = []
    iou_dist = []

    for image_patches, target_patches in data_loader:
        for image, target in zip(image_patches, target_patches):
            if target["boxes"].numel() == 0:
                continue

            with torch.no_grad():
                prediction = model([image.to(device)])[0]

            # Prepare predictions and targets for torchmetrics
            preds = {
                "boxes": prediction["boxes"].cpu(),
                "scores": prediction["scores"].cpu(),
                "labels": prediction["labels"].cpu()
            }

            targets = {
                "boxes": target["boxes"].cpu(),
                "labels": target["labels"].cpu()
            }
            pred_boxes = prediction["boxes"].cpu().numpy()
            true_boxes = target["boxes"].cpu().numpy()

            for pred_box in pred_boxes:
                for true_box in true_boxes:
                    iou = calculate_iou(pred_box, true_box)
                    iou_dist.append(iou)

            # Update the metric for the overall dataset
            overall_metric.update([preds], [targets])
            recall_50.update([preds], [targets])

            # Calculate the metric for this specific patch
            patch_metric = MeanAveragePrecision(class_metrics=True)
            patch_metric.update([preds], [targets])
            patch_result = patch_metric.compute()
            patch_metrics.append(patch_result)

    # Compute the final mAP and other metrics for the whole dataset
    overall_result = overall_metric.compute()
    recall_50_result = recall_50.compute()

    recall_at_iou_50 = recall_50_result['mar_100_per_class']


    print(f"Overall mAP: {overall_result['map']:.4f}")
    print(f"Overall mAP_50: {overall_result['map_50']:.4f}")
    print(f"Overall mAP_75: {overall_result['map_75']:.4f}")
    print(f"Precision per class: {overall_result['map_per_class']}")
    print(f"Recall per class: {overall_result['mar_100_per_class']}")
    print(f'Recall when IoU >= 50: {recall_at_iou_50}')
    #print(f'Recall at mAP_50: {overall_result['mar_100_iou_0.50']}')
    #print(f"mAP per class: {overall_result['map_per_class']}")

    # Optionally plot metrics for individual patches
    
    iou_dist_50 = [iou for iou in iou_dist if iou >= 0.5]
    plot_iou_histogram_50(iou_dist_50)
    print(f'Mean IoU >= 50: {np.mean(iou_dist_50):.4f}')

    plot_iou_histogram(iou_dist)
    print(f'Mean IoU: {np.mean(iou_dist):.4f}')

    plot_patch_metrics(patch_metrics, overall_result)
    return overall_result, patch_metrics

def plot_patch_metrics(patch_metrics, overall_result):
    maps = [metric['map'].cpu().item() for metric in patch_metrics]
    map_50s = [metric['map_50'].cpu().item() for metric in patch_metrics]
    recalls = [metric['mar_100_per_class'].cpu().item() for metric in patch_metrics]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(maps, bins=10, color='blue', alpha=0.7)
    plt.axvline(x=overall_result['map'].cpu().item(), color='black', linestyle='--', label="Overall mAP")
    plt.xlabel('mAP Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of mAP per Patch')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.hist(map_50s, bins=10, color='green', alpha=0.7)


    plt.subplot(1, 3, 3)
    plt.hist(recalls, bins=10, color='purple', alpha=0.7)
    plt.axvline(x=overall_result['mar_100_per_class'].mean().cpu().item(), color='black', linestyle='--', label="Overall Recall")
    plt.xlabel('Recall Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Recall per Patch')
    plt.legend()

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    granzyme_b_image_folder = '/UEC, CD8 and GranzymeB'
    labels_folder = '/Lables/manual_box_label'
    patch_size = 256


    dataset_val = GranzymeBDataset(granzyme_b_image_folder, labels_folder, transform=get_val_transforms(), patch_size=patch_size, val=True)
    data_val_loader = DataLoader(dataset_val, batch_size=2, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model(num_classes=2)  # 2 classes: background and Granzyme B
    model.to(device)

    load_path = 'data/saved_models/new_15epochs_model.pth'

    model.load_state_dict(torch.load(load_path))


    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005) #lr = 0.0005 for 10 epochs fine tuning
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    model_evaluation(model, data_val_loader, device)
