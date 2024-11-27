import dropbox
import io
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
import json
import torch
import torchvision.transforms.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.patches as mpatches
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

# Dropbox access token
ACCESS_TOKEN = 'sl.B6Qd0mVRW3wsUNThQScSR8_N74KTT5BzWvV-cruXmReot1KfVpyGnk4e6W3B_M8g3d7ib_PkLz8nNz-CUq0-mwHeX0ShjnrgZ0_xh856M6gbwu58cPjC7NFRvtRqnuf4ODoJTa-q-K9j'
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

# Plot image with bounding boxes
def plot_image_with_boxes(image, boxes, pred_boxes=None, pred_scores=None ,title="Image with Bounding Boxes"):
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)
    for box in boxes:
        bb = box["Bounding Box"]
        rect = mpatches.Rectangle(
            (bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1],
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(bb[0], bb[1] - 10, f"ID: {box['Cell ID']}, GB: {box['Granzyme B']:.2f}", color='yellow', fontsize=8, backgroundcolor="none")
    if pred_boxes is not None:
       # for pred_box in pred_boxes:
        for pred_box, score in zip(pred_boxes, pred_scores):
            rect = mpatches.Rectangle(
                (pred_box[0], pred_box[1]), pred_box[2] - pred_box[0], pred_box[3] - pred_box[1],
                linewidth=2, edgecolor='blue', facecolor='none', linestyle='dashed'
            )
            ax.add_patch(rect)
            ax.text(pred_box[0], pred_box[1] - 10, f'{score:.2f}', 
                    color='black', fontsize=12, verticalalignment='top', 
                    bbox=dict(facecolor='white', alpha=0.5))
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
                bb = box["Bounding Box"]
                if (bb[0] >= j and bb[2] <= j + patch_size and
                        bb[1] >= i and bb[3] <= i + patch_size):
                    adjusted_box = {
                        "Bounding Box": [
                            bb[0] - j,
                            bb[1] - i,
                            bb[2] - j,
                            bb[3] - i
                        ],
                        "Cell ID": box["Cell ID"],
                        "Granzyme B": box["Granzyme B"]
                    }
                    patch_boxes.append(adjusted_box)
            if patch_boxes:
                patches.append((patch, patch_boxes, i, j))

    return patches

# Model initialization

def get_model(num_classes: int):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    model.roi_heads.mask_predictor = None
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Run inference and visualize results
def run_inference_and_update_patches(model, patches, device):
    model.eval()
    updated_patches = []
    for patch, patch_boxes, i, j in patches:
        patch_tensor = F.to_tensor(patch).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(patch_tensor)[0]
        pred_boxes = prediction['boxes'].cpu().numpy()
        pred_scores = prediction['scores'].cpu().numpy()
        
        updated_patches.append((patch, patch_boxes, pred_boxes, pred_scores, i, j))
        plot_image_with_boxes(patch, patch_boxes, pred_boxes,pred_scores ,title=f"Patch ({i}, {j}) with Bounding Boxes")
    return updated_patches

# Reconstruct the original image from patches
def reconstruct_image(original_image, patches, patch_size):
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(original_image)
    for patch, patch_boxes, pred_boxes,pred_scores, i, j in patches:
        for pred_box in pred_boxes:
            x_min, y_min, x_max, y_max = pred_box
            rect = mpatches.Rectangle(
                (x_min + j, y_min + i), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor='blue', facecolor='none', linestyle='dashed'
            )
            ax.add_patch(rect)
            #ax.text(x_min + j, y_min + i, f'{pred_scores:.2f}', color='black', fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        for patch_box in patch_boxes:
            bb = patch_box["Bounding Box"]
            rect = mpatches.Rectangle(
                (bb[0] + j, bb[1] + i), bb[2] - bb[0], bb[3] - bb[1],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            #ax.text(bb[0] + j, bb[1] + i - 10, f"ID: {patch_box['Cell ID']}, GB: {patch_box['Granzyme B']:.2f}", color='yellow', fontsize=8, backgroundcolor="none")
        original_image[i:i + patch_size, j:j + patch_size] = patch
    return original_image

# Main function to process and plot images with bounding boxes and patches
def main(image_folder, labels_folder, patch_size, model):
    image_files = list_files_in_folder(image_folder)
    label_files = list_files_in_folder(labels_folder)

    label_files = label_files[:4]
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #model = get_model(num_classes=2)  # 2 classes: background and Granzyme B
    #model.to(device)

    for label_file in label_files:
        if label_file.endswith('_labels.json'):
            sample_name = label_file.replace('_labels.json', '')
            img_name = sample_name + '_Granzyme B_path_view.tif'
            if img_name in image_files:
                # Load image and annotation
                image_path = f"{image_folder}/{img_name}"
                label_path = f"{labels_folder}/{label_file}"

                image = load_image(image_path)
                annotation = load_annotation(label_path)

                # Filter boxes for Granzyme B
                granzyme_b_boxes = [box for box in annotation if "Granzyme B" in box]

                # Plot full image with bounding boxes
                plot_image_with_boxes(image, granzyme_b_boxes, title="Full Image with Bounding Boxes")

                # Create patches and plot each patch with bounding boxes
                patches = create_patches(image, granzyme_b_boxes, patch_size)

                # Run inference and update patches with predictions
                updated_patches = run_inference_and_update_patches(model, patches, device)

                # Reconstruct the image
                reconstructed_image = reconstruct_image(image, updated_patches, patch_size)

                # Plot the reconstructed image
                plt.imshow(reconstructed_image)
                plt.title("Reconstructed Image with Bounding Boxes")
                plt.axis('off')
                plt.tight_layout()
                plt.show()


if __name__=='__main__':
    image_folder = '/UEC, CD8 and GranzymeB'
    labels_folder = '/Lables/new_json_test'
    patch_size = 256
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model(num_classes=2)
    model.to(device)
    model.load_state_dict(torch.load('new_15epochs_model.pth'))
    #load_path = "new_15epochs_model.pth"
    #model.load_state_dict(torch.load(load_path))
    

    main(image_folder, labels_folder, patch_size, model)
