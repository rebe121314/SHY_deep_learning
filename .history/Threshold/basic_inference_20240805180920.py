import io
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as mpatches
from skimage import io as skio, img_as_ubyte, measure
from skimage.color import rgb2hed, hed2rgb
from skimage.exposure import rescale_intensity, equalize_adapthist
from cellpose import models
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from matplotlib.widgets import Button
from mpl_interactions import panhandler, zoom_factory
from config import ACCESS_TOKEN
from utils.dropbox_utils import DropboxHandler
from utils.image_utils import ImageProcessor
from utils.data_utils import global_inform_values, label_relevant_cells

# Dropbox access token
dbx = DropboxHandler(ACCESS_TOKEN)

class Inference:
    def __init__(self, inform_files, granzyme_b_image_folder, model_path, patch_size, manual=True):
        self.inform_files = inform_files
        self.granzyme_b_image_folder = granzyme_b_image_folder
        self.errors = []
        self.model_path = model_path
        self.patch_size = patch_size
        self.manual = manual

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.get_model(num_classes=2)  # 2 classes: background and Granzyme B
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.rl_env = RLEnvironment(self.model, self.device, patch_size)

    def get_model(self, num_classes):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
        model.roi_heads.mask_predictor = None
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def list_files_in_folder(self, folder_path):
        return dbx.list_files_in_folder(folder_path)

    def read_image_from_dropbox(self, dropbox_path):
        return dbx.read_image_from_dropbox(dropbox_path)

    def read_excel_from_dropbox(self, dropbox_path):
        return dbx.read_excel_from_dropbox(dropbox_path)

    def relevant_rows(self, image_name, inform_excel):
        relevant_cells = pd.DataFrame()
        for index, row in inform_excel.iterrows():
            if image_name in row['Sample Name']:
                relevant_cells = relevant_cells._append(row)
        return relevant_cells

    def create_patches_no_box(self, image, patch_size):
        patches = []
        img_height, img_width = image.shape[:2]
        for i in range(0, img_height, patch_size):
            for j in range(0, img_width, patch_size):
                patch = image[i:i + patch_size, j:j + patch_size]
                patches.append((patch, i, j))
        return patches

    def run_inference_on_patches(self, patches):
        updated_patches = []
        self.model.eval()
        for patch, i, j in patches:
            patch_tensor = F.to_tensor(patch).unsqueeze(0).to(self.device)
            with torch.no_grad():
                prediction = self.model(patch_tensor)[0]
            pred_boxes = prediction['boxes'].cpu().numpy()
            pred_scores = prediction['scores'].cpu().numpy()
            updated_patches.append((patch, pred_boxes, pred_scores, i, j))
        return updated_patches

    def generate_bounding_boxes(self, patch, properties, pred_boxes, pred_scores, criteria_labels, i_offset, j_offset):
        boxes = []
        margin = 5  # Large margin to accommodate the possible bounding boxes
        score_threshold = 0.5

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

    def full_processing(self, image, patch_size, criteria_labels, image_name):
        patches = self.create_patches_no_box(image, patch_size)
        patches = self.run_inference_on_patches(patches)
        patches_with_boxes = []

        for patch, pred_boxes, pred_scores, i, j in patches:
            properties = self.process_image(patch)
            boxes = self.generate_bounding_boxes(patch, properties, pred_boxes, pred_scores, criteria_labels, i, j)
            patches_with_boxes.append((patch, boxes, i, j))

        all_boxes = [box for _, boxes, _, _ in patches_with_boxes for box in boxes]

        if self.manual:
            valid_boxes = self.manual_elimination(image, all_boxes)
            self.reinforcement_learning(image, valid_boxes, num_epochs=3)
            return valid_boxes

        new_image = self.reconstruct_image(image, patches_with_boxes, patch_size)
        return all_boxes

    def reconstruct_image(self, original_image, patches_with_boxes, patch_size):
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(original_image)
        for patch, patch_boxes, i, j in patches_with_boxes:
            for patch_box in patch_boxes:
                bb = patch_box["Bounding Box"]
                x_min, y_min, x_max, y_max = bb
                rect = mpatches.Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x_min, y_min, f'{patch_box["Score"]:.2f}', color='black', fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
            original_image[i:i + patch_size, j:j + patch_size] = patch
        return original_image

    def plot_image_with_boxes(self, image, boxes, pred_boxes=None, pred_scores=None, title="Image with Bounding Boxes"):
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(image)
        for box in boxes:
            bb = box["Bounding Box"]
            rect = mpatches.Rectangle(
                (bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(bb[0], bb[1] - 20, f'{box["Score"]:.2f}', color='black', fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        if pred_boxes is not None:
            for pred_box, score in zip(pred_boxes, pred_scores):
                rect = mpatches.Rectangle(
                    (pred_box[0], pred_box[1]), pred_box[2] - pred_box[0], pred_box[3] - pred_box[1],
                    linewidth=2, edgecolor='blue', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(pred_box[0], pred_box[1] - 10, f'{score:.2f}', color='black', fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def manual_elimination(self, image, boxes):
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(image)
        patches = []
        texts = []
        selected = [False] * len(boxes)  # Track selection status of each box

        for box in boxes:
            bb = box["Bounding Box"]
            rect = mpatches.Rectangle(
                (bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1],
                linewidth=2, edgecolor='blue', facecolor='none'
            )
            patches.append(rect)
            ax.add_patch(rect)
            text = ax.text(bb[0], bb[1] - 20, f'{box["Score"]:.2f}', color='black', fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
            texts.append(text)
        plt.title('Manual Elimination of Bounding Boxes')

        def on_click(event):
            if event.inaxes != ax:
                return
            for i, (box, rect, text) in enumerate(zip(boxes, patches, texts)):
                bb = box["Bounding Box"]
                x_min, y_min, x_max, y_max = bb
                if x_min < event.xdata < x_max and y_min < event.ydata < y_max:
                    selected[i] = not selected[i]
                    if selected[i]:
                        rect.set_edgecolor('red')
                        rect.set_linestyle('dotted')
                        text.set_visible(False)
                    else:
                        rect.set_edgecolor('blue')
                        rect.set_linestyle('solid')
                        text.set_visible(True)
                    fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', on_click)

        def on_done(event):
            plt.close(fig)

        ax_button = plt.axes([0.9, 0.0, 0.1, 0.075])
        button = Button(ax_button, 'Done')
        button.on_clicked(on_done)

        plt.axis('off')
        plt.tight_layout()

        # Enable smooth zoom and pan functionality
        zoom_factory(ax)
        panhandler(fig)
        fig.canvas.manager.toolbar.zoom()

        plt.show()

        # Update boxes list based on selected status
        updated_boxes = [box for i, box in enumerate(boxes) if not selected[i]]

        return updated_boxes

    def reinforcement_learning(self, image, true_boxes, num_epochs=3):
        for epoch in range(num_epochs):
            state = self.rl_env.reset(image)
            done = False
            while not done:
                patch, i, j = state
                patch_tensor = F.to_tensor(patch).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    prediction = self.model(patch_tensor)[0]
                pred_boxes = prediction['boxes'].cpu().numpy()
                pred_scores = prediction['scores'].cpu().numpy()

                true_boxes_coords = [box['Bounding Box'] for box in true_boxes]
                valid_pred_boxes = [box for box in pred_boxes if self.rl_env.is_valid_box(box)]

                action = valid_pred_boxes
                next_state, reward, done = self.rl_env.step(action, valid_pred_boxes, pred_scores, true_boxes_coords)
                state = next_state

    def color_separate(self, ihc_rgb):
        ihc_hed = rgb2hed(ihc_rgb)
        null = np.zeros_like(ihc_hed[:, :, 0])
        ihc_h = img_as_ubyte(hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1)))
        ihc_e = img_as_ubyte(hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1)))
        ihc_d = img_as_ubyte(hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1)))
        h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1), in_range=(0, np.percentile(ihc_hed[:, :, 0], 99)))
        d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1), in_range=(0, np.percentile(ihc_hed[:, :, 2], 99)))
        zdh = img_as_ubyte(np.dstack((null, d, h)))
        return ihc_h, ihc_e, ihc_d, zdh

    def process_image(self, gb):
        H, _, D, _ = self.color_separate(gb)
        hematoxylin_eq = equalize_adapthist(H[:, :, 0])
        input_image = 1 - hematoxylin_eq

        model = models.Cellpose(gpu=True, model_type='nuclei')
        masks, flows, styles, diams = model.eval(input_image, diameter=None, channels=[0, 0])

        segmented_np = masks
        properties = measure.regionprops(segmented_np, intensity_image=H[:, :, 0])

        nuclei_intensities = []
        dab_intensities = []
        circularities = []
        for prop in properties:
            minr, minc, maxr, maxc = prop.bbox
            cell_mask = masks[minr:maxr, minc:maxc] == prop.label
            nuclei_intensity = np.mean(H[minr:maxr, minc:maxc][cell_mask])
            dab_intensity = np.mean(D[minr:maxr, minc:maxc][cell_mask])
            nuclei_intensities.append(nuclei_intensity)
            dab_intensities.append(dab_intensity)

            area = prop.area
            perimeter = prop.perimeter
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
            circularities.append(circularity)

        intensity_threshold = np.percentile(nuclei_intensities, 10)

        lymphocytes_with_dab = [
            prop.label for prop, nuclei_intensity, dab_intensity, circularity in zip(properties, nuclei_intensities, dab_intensities, circularities)
            if nuclei_intensity < intensity_threshold and dab_intensity > 0 and circularity > 0.55
        ]

        lymphocyte_dab_mask = np.isin(masks, lymphocytes_with_dab)
        filtered_properties = [prop for prop in properties if prop.label in lymphocytes_with_dab]
        return filtered_properties

    def run(self, example):
        inform_files = self.list_files_in_folder(self.inform_files)
        image_files = self.list_files_in_folder(self.granzyme_b_image_folder)

        for inform_file in inform_files:
            if inform_file.endswith('.xlsx'):
                inform_path = self.inform_files + inform_file
                inform_excel = self.read_excel_from_dropbox(inform_path)
                name_sample = inform_file.split('.xlsx')[0]
                if name_sample != example:
                    continue
                auto_98 = global_inform_values(inform_excel)
                print('Working on sample:', name_sample)

                relevant_images = [f for f in image_files if name_sample in f and 'Granzyme' in f]
                print('Number of images in sample:', len(relevant_images))
                
                valid_cells = []

                for image_file in relevant_images:
                    try:
                        image_path = self.granzyme_b_image_folder + image_file
                        image_name = image_file.split('_Granzyme')[0]
                        print('Working on image:', image_name)
                        relevant_rows = self.relevant_rows(image_name, inform_excel)
                        criteria_labels = label_relevant_cells(relevant_rows, auto_98)
                        gb_image = self.read_image_from_dropbox(image_path)
                        patch_size = 256
                        all_boxes = self.full_processing(gb_image, patch_size, criteria_labels, image_name)
                        valid_cells.append(all_boxes)
                    except Exception as e:
                        self.errors.append({'image': image_file, 'error': str(e)})
                        print(f'Error processing {image_file}: {e}')
                        json_path = f'new_simpie_flexible/error/{image_name}_error.json'
                        with open(json_path, 'w') as json_file:
                            json.dump(image_name, json_file)
                print('Done!')
                return valid_cells

# Folder paths
inform_files = '/Rebeca&Laura/inform_in_excel/'
granzyme_b_image_folder = '/UEC, CD8 and GranzymeB/'
model_path = 'new_15epochs_model.pth'
patch_size = 256
example_sample = 'Opal 221_8'  # Example

# Example usage:
inference = Inference(inform_files, granzyme_b_image_folder, model_path, patch_size)
valid_cells = inference.run(example_sample)
print(valid_cells)
