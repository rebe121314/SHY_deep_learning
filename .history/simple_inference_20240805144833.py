# simple_inference.py

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
from config import DROPBOX_ACCESS

dbx = dropbox.Dropbox(DROPBOX_ACCESS)

def list_files_in_folder(folder_path):
    files = []
    result = dbx.files_list_folder(folder_path)
    while True:
        files.extend([entry.name for entry in result.entries if isinstance(entry, dropbox.files.FileMetadata)])
        if not result.has_more:
            break
        result = dbx.files_list_folder_continue(result.cursor)
    return files

def load_annotation(dropbox_path):
    _, res = dbx.files_download(dropbox_path)
    return json.load(io.BytesIO(res.content))

def load_image(dropbox_path):
    _, res = dbx.files_download(dropbox_path)
    file_bytes = io.BytesIO(res.content)
    return skio.imread(file_bytes)

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

def plot_image_with_boxes(image, boxes, pred_boxes=None, pred_scores=None, title="Image with Bounding Boxes"):
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

def get_model(num_classes: int):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    model.roi_heads.mask_predictor = None
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def run_inference_on_patches(model, patches, device):
    updated_patches = []
    model.eval()
    for patch, i, j in patches:
        patch_tensor = F.to_tensor(patch).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(patch_tensor)[0]
        pred_boxes = prediction['boxes'].cpu().numpy()
        pred_scores = prediction['scores'].cpu().numpy()
        updated_patches.append((patch, pred_boxes, pred_scores, i, j))
    return updated_patches

def full_processing(image, patch_size, criteria_labels, image_name, model, device):
    patches = create_patches(image, criteria_labels, patch_size)
    patches = run_inference_on_patches(model, patches, device)
    patches_with_boxes = []

    for patch, pred_boxes, pred_scores, i, j in patches:
        boxes = generate_bounding_boxes(patch, pred_boxes, pred_scores, criteria_labels, i, j)
        patches_with_boxes.append((patch, boxes, i, j))

    all_boxes = [box for _, boxes, _, _ in patches_with_boxes for box in boxes]
    return all_boxes

def generate_bounding_boxes(patch, pred_boxes, pred_scores, criteria_labels, i_offset, j_offset, score_threshold=0.75):
    boxes = []
    margin = 5
    for pred_box, score in zip(pred_boxes, pred_scores):
        if score < score_threshold:
            continue
        x_min, y_min, x_max, y_max = pred_box
        expanded_minc = max(0, x_min - margin) + j_offset
        expanded_maxc = x_max + margin + j_offset
        expanded_minr = max(0, y_min - margin) + i_offset
        expanded_maxr = y_max + margin + i_offset
        for _, row in criteria_labels.iterrows():
            x_position = row['Cell X Position']
            y_position = row['Cell Y Position']
            if expanded_minc <= x_position <= expanded_maxc and expanded_minr <= y_position <= expanded_maxr:
                boxes.append({
                    'Cell ID': row['Cell ID'],
                    'X Position': int(x_position),
                    'Y Position': int(y_position),
                    'Bounding Box': [int(x_min + j_offset), int(y_min + i_offset), int(x_max + j_offset), int(y_max + i_offset)],
                    'Granzyme B': row['Entire Cell Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)'],
                    'Score': score
                })
    return boxes

def main():
    inform_files = '/Rebeca&Laura/inform_in_excel/'
    granzyme_b_image_folder = '/UEC, CD8 and GranzymeB/'
    model_path = 'new_15epochs_model.pth'
    patch_size = 256

    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(model_path))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    inform_files_list = list_files_in_folder(inform_files)
    image_files = list_files_in_folder(granzyme_b_image_folder)

    for inform_file in inform_files_list:
        if inform_file.endswith('.xlsx'):
            inform_path = inform_files + inform_file
            inform_excel = read_excel_from_dropbox(inform_path)
            name_sample = inform_file.split('.xlsx')[0]

            relevant_images = [f for f in image_files if name_sample in f and 'Granzyme' in f]

            for image_file in relevant_images:
                image_name = image_file.split('_Granzyme')[0]
                relevant_rows = relevant_rows(image_name, inform_excel)
                criteria_labels = label_relevant_cells(relevant_rows)
                gb_image = load_image(granzyme_b_image_folder + image_file)
                all_boxes = full_processing(gb_image, patch_size, criteria_labels, image_name, model, device)
                plot_image_with_boxes(gb_image, all_boxes, title=f"{image_name} - Inference")

if __name__ == "__main__":
    main()
