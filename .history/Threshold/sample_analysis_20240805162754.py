import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import seaborn as sns
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from mpl_interactions import panhandler, zoom_factory
from cellpose import models
from skimage import io as skio, measure, img_as_ubyte
from skimage.color import rgb2hed, hed2rgb
from skimage.exposure import rescale_intensity, equalize_adapthist
from tqdm import tqdm
import pickle
from config import ACCESS_TOKEN
from utils.dropbox_utils import DropboxHandler
from utils.data_utils import global_inform_values, label_relevant_cells

# Configurations
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
            exclude_boxes = [box for box in all_boxes if box not in valid_boxes]
            return valid_boxes

        return all_boxes

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
        inform_files = dbx.list_files_in_folder(self.inform_files)
        image_files = dbx.list_files_in_folder(self.granzyme_b_image_folder)
        errors = os.listdir('/Users/rebeca/Documents/Code/SHY_lab/GB_Deep/processed_data/errors')
        errors = [f.split('_error.json')[0] for f in errors]

        for inform_file in inform_files:
            if inform_file.endswith('.xlsx'):
                inform_path = self.inform_files + inform_file
                inform_excel = dbx.read_excel_from_dropbox(inform_path)
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

                    if f'{image_name}' in errors:
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
                            gb_image = dbx.read_image_from_dropbox(image_path)
                            all_boxes = self.full_processing(gb_image, self.patch_size, criteria_labels, image_name)
                            valid_cells.extend(all_boxes)
                            self.save_processed_data(name_sample, image_name, all_boxes)
                            print(f'The image {image_name} was saved')
                        except Exception as e:
                            self.errors.append({'image': image_file, 'error': str(e)})
                            print(f'Error processing {image_file}: {e}')
                            json_path = f'processed_data/errors/{image_name}_error.json'
                            with open(json_path, 'w') as json_file:
                                json.dump(image_name, json_file)

                print(f'Done processing sample: {name_sample}')
                optimal_threshold, inform_data = self.calculate_threshold(valid_cells, inform_excel)
                return optimal_threshold, inform_data

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
        return optimal_threshold, inform_data


class GranzymeBAnalysis:
    def __init__(self, data, threshold):
        self.data = data
        self.threshold = threshold
        self.target_cell1 = "CD8"
        self.target_cell2 = "CD4"
        self.target_cell3 = "CD56"
        self.target_cell4 = "Tumor cell"
        self.marker = "Granzyme B"
        self.rename_columns()
        self.preprocess_data()

    def rename_columns(self):
        self.data.rename(columns={
            'Entire Cell Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)': 'GranzymeB',
            'Nucleus CD8 (Opal 570) Mean (Normalized Counts, Total Weighting)': 'CD8_Nuc',
            'Cytoplasm CD8 (Opal 570) Mean (Normalized Counts, Total Weighting)': 'CD8_Cyt',
            'Membrane CD8 (Opal 570) Mean (Normalized Counts, Total Weighting)': 'CD8_Mem',
            'Nucleus CD4 (Opal 520) Mean (Normalized Counts, Total Weighting)': 'CD4',
            'Nucleus CD56 (Opal 540) Mean (Normalized Counts, Total Weighting)': 'CD56',
            'Nucleus Nucleus (DAPI) Mean (Normalized Counts, Total Weighting)': 'DAPI',
            'Nucleus Autofluorescence Mean (Normalized Counts, Total Weighting)': 'Auto'
        }, inplace=True)

    def preprocess_data(self):
        relevant_columns = [
            'GranzymeB', 'CD8_Nuc', 'CD8_Cyt', 'CD8_Mem', 'CD4', 'CD56', 'DAPI', 'Auto'
        ]

        self.data = self.data[relevant_columns + ['True Label', 'Phenotype']]
        self.data[relevant_columns] = self.data[relevant_columns].apply(pd.to_numeric, errors='coerce')
        self.data.dropna(inplace=True)

        scaler = MinMaxScaler()
        self.data[relevant_columns] = scaler.fit_transform(self.data[relevant_columns]) + 0.000001

    def visualize_granzyme_b_distribution(self):
        sns.histplot(self.data['GranzymeB'], kde=True)
        plt.axvline(self.threshold, color='r', linestyle='--')
        plt.title('Distribution of Granzyme B Mean Intensity')
        plt.xlabel('Granzyme B Mean Intensity')
        plt.ylabel('Frequency')
        plt.show()

    def train_classifier(self):
        X = self.data.drop(columns=['True Label', 'Phenotype'])
        y = self.data['True Label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        scores = cross_val_score(model, X, y, cv=5)
        mean_score = scores.mean()
        std_score = scores.std()

        print(f'Model Accuracy: {accuracy}')
        print(f'Mean Cross-Validation Score: {mean_score}')
        print(f'Standard Deviation of Cross-Validation Scores: {std_score}')

        return model, mean_score, std_score

    def classify_and_visualize(self, model):
        self.data['Predicted Label'] = model.predict(self.data.drop(columns=['True Label', 'Phenotype']))

        target_labels = [self.target_cell1, self.target_cell2, self.target_cell3, self.target_cell4]

        for target_label in target_labels:
            target_data = self.data[self.data['Phenotype'] == target_label]

            fig = plt.figure(figsize=(8, 6)) 
            plt.scatter(target_data['CD8_Nuc'], target_data['GranzymeB'], s=0.1, c=(target_data['GranzymeB'] >= self.threshold).astype(int), cmap='coolwarm')
            plt.xlim(0.00005, 5)
            plt.ylim(0.0000005, 5)
            plt.axhline(y=self.threshold, color='r', linestyle='dashed')
            plt.xscale("log")
            plt.yscale("log")
            plt.ylabel('Granzyme B Mean Intensity', fontsize=25)
            plt.xlabel('CD8 Mean Intensity', fontsize=25)
            plt.title(f'For {target_label} cell type', fontsize=30)
            
            CD8Y = target_data['CD8_Nuc']
            Quadrant_1 = target_data[(CD8Y >= self.threshold)].shape[0] / target_data.shape[0]
            Quadrant_4 = target_data[(CD8Y < self.threshold)].shape[0] / target_data.shape[0]
            plt.text(0.5, 1, '{:.1%}'.format(Quadrant_1), fontsize=25)
            plt.text(0.5, 0.000001, '{:.1%}'.format(Quadrant_4), fontsize=25)
            
            plt.show()

            fig = plt.figure(figsize=(8, 6)) 
            plt.scatter(target_data['DAPI'], target_data['GranzymeB'], s=0.1, c=(target_data['GranzymeB'] >= self.threshold).astype(int), cmap='coolwarm')
            plt.xlim(0.00005, 5)
            plt.ylim(0.0000005, 5)
            plt.axhline(y=self.threshold, color='r', linestyle='dashed')
            plt.xscale("log")
            plt.yscale("log")
            plt.ylabel('Granzyme B Mean Intensity', fontsize=25)
            plt.xlabel('DAPI Mean Intensity', fontsize=25)
            plt.title(f'For {target_label} cell type', fontsize=30)
            
            DAPIY = target_data['DAPI']
            Quadrant_1 = target_data[(DAPIY >= self.threshold)].shape[0] / target_data.shape[0]
            Quadrant_4 = target_data[(DAPIY < self.threshold)].shape[0] / target_data.shape[0]
            plt.text(0.5, 1, '{:.1%}'.format(Quadrant_1), fontsize=25)
            plt.text(0.5, 0.000001, '{:.1%}'.format(Quadrant_4), fontsize=25)
            
            plt.show()

    def run(self):
        print("Starting Granzyme B analysis...")
        print("Visualizing Granzyme B distribution...")
        self.visualize_granzyme_b_distribution()
        print("Training classifier...")
        model, mean_score, std_score = self.train_classifier()
        print("Classifying and visualizing data...")
        self.classify_and_visualize(model)

# Example usage:
inform_files = '/Rebeca&Laura/inform_in_excel/'
granzyme_b_image_folder = '/UEC, CD8 and GranzymeB/'
processed_data_folder = 'processed_data'
model_path = 'new_15epochs_model.pth'
patch_size = 256
example_sample = 'Opal 221_8'  # Example

inference = InferenceWithThreshold(inform_files, granzyme_b_image_folder, processed_data_folder, model_path, patch_size)
optimal_threshold, inform_data = inference.run(example_sample)

granzyme_b_analysis = GranzymeBAnalysis(inform_data, optimal_threshold)
granzyme_b_analysis.run()

