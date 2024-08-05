import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from mpl_interactions import panhandler, zoom_factory
import pickle
from sklearn.metrics import roc_curve, auc

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import ACCESS_TOKEN, INFORM_FILES_PATH, GRANZYME_B_IMAGE_FOLDER, PROCESSED_DATA_FOLDER, MODEL_PATH, PATCH_SIZE
from utils.dropbox_utils import DropboxHandler
from utils.data_utils import global_inform_values, label_relevant_cells
from utils.image_utils import process_image

dbx = DropboxHandler(ACCESS_TOKEN)

class InferenceWithThreshold:
    def __init__(self, inform_files, granzyme_b_image_folder, processed_data_folder, model_path, patch_size, manual=False):
        self.inform_files = inform_files
        self.granzyme_b_image_folder = granzyme_b_image_folder
        self.processed_data_folder = processed_data_folder
        self.errors = []
        self.model_path = model_path
        self.patch_size = patch_size
        self.manual = manual

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.get_model(num_classes=2)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

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
        margin = 5
        score_threshold = 0.75

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
            properties = process_image(patch)
            boxes = self.generate_bounding_boxes(patch, properties, pred_boxes, pred_scores, criteria_labels, i, j)
            patches_with_boxes.append((patch, boxes, i, j))

        all_boxes = [box for _, boxes, _, _ in patches_with_boxes for box in boxes]

        if self.manual:
            valid_boxes = self.manual_elimination(image, all_boxes)
            return valid_boxes

        return all_boxes

    def manual_elimination(self, image, boxes):
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(image)
        patches = []
        texts = []
        selected = [False] * len(boxes)

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
        zoom_factory(ax)
        panhandler(fig)
        fig.canvas.manager.toolbar.zoom()
        plt.show()

        updated_boxes = [box for i, box in enumerate(boxes) if not selected[i]]
        return updated_boxes

    def save_processed_data(self, sample_name, image_name, data):
        sample_folder = os.path.join(self.processed_data_folder, sample_name)
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
        file_path = os.path.join(sample_folder, f'{image_name}.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def load_processed_data(self, sample_name, image_name):
        file_path = os.path.join(self.processed_data_folder, sample_name, f'{image_name}.pkl')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        return None

    def run(self, example):
        inform_files = self.list_files_in_folder(self.inform_files)
        image_files = self.list_files_in_folder(self.granzyme_b_image_folder)
        errors = os.listdir(os.path.join(self.processed_data_folder, 'errors'))
        errors = [f.split('_error.json')[0] for f in errors]

        for inform_file in inform_files:
            if inform_file.endswith('.xlsx'):
                inform_path = self.inform_files + inform_file
                inform_excel = self.read_excel_from_dropbox(inform_path)
                name_sample = inform_file.split('.xlsx')[0]
                if name_sample != example:
                    continue

                print(f'Working on sample: {name_sample}')
                relevant_images = [f for f in image_files if name_sample in f and 'Granzyme' in f]
                print(f'Number of images in sample: {len(relevant_images)}')

                valid_cells = []
                processed_sample_folder = os.path.join(self.processed_data_folder, name_sample)
                if not os.path.exists(processed_sample_folder):
                    os.makedirs(processed_sample_folder)

                for image_file in relevant_images:
                    image_name = image_file.split('_Granzyme')[0]

                    if image_name in errors:
                        print(f'Skipping image {image_name} due to previous error')
                        continue

                    processed_data = self.load_processed_data(name_sample, image_name)
                    if processed_data:
                        print(f'The image {image_name} was already processed!')
                        valid_cells.extend(processed_data)
                    else:
                        try:
                            image_path = self.granzyme_b_image_folder + image_file
                            print(f'Processing image: {image_name}')
                            relevant_rows = self.relevant_rows(image_name, inform_excel)
                            criteria_labels = label_relevant_cells(relevant_rows)
                            gb_image = self.read_image_from_dropbox(image_path)
                            all_boxes = self.full_processing(gb_image, self.patch_size, criteria_labels, image_name)
                            valid_cells.extend(all_boxes)
                            self.save_processed_data(name_sample, image_name, all_boxes)
                            print(f'The image {image_name} was saved')
                        except Exception as e:
                            self.errors.append({'image': image_file, 'error': str(e)})
                            print(f'Error processing {image_file}: {e}')
                            json_path = os.path.join(self.processed_data_folder, 'errors', f'{image_name}_error.json')
                            with open(json_path, 'w') as json_file:
                                json.dump(image_name, json_file)

                print(f'Done processing sample: {name_sample}')
                optimal_threshold = self.calculate_threshold(valid_cells, inform_excel)
                return optimal_threshold, inform_excel

    def calculate_threshold(self, valid_cells, inform_data):
        valid_cell_ids = [cell['Cell ID'] for cell in valid_cells]
        inform_data['True Label'] = inform_data['Cell ID'].apply(lambda x: 1 if x in valid_cell_ids else 0)

        true_labels = inform_data['True Label'].tolist()
        granzyme_b_values = inform_data['Entire Cell Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)'].tolist()

        fpr, tpr, thresholds = roc_curve(true_labels, granzyme_b_values)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='purple', label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()

        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print(f'Optimal threshold for Granzyme B: {optimal_threshold}')
        return optimal_threshold

# Example usage:
example = 'Opal 221_8'
process_data_folder = PROCESSED_DATA_FOLDER + '/' + example
inference = InferenceWithThreshold(INFORM_FILES_PATH, GRANZYME_B_IMAGE_FOLDER, PROCESSED_DATA_FOLDER, MODEL_PATH, PATCH_SIZE)
optimal_threshold, inform_data = inference.run(example)
