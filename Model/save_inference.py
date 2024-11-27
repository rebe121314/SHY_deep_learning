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
import dropbox
import io
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import albumentations as A
from skimage import io as skio, img_as_ubyte, measure
from skimage.color import rgb2hed, hed2rgb
from skimage.exposure import rescale_intensity, equalize_adapthist
import json
from tqdm import tqdm
import matplotlib.patches as mpatches
from cellpose import models
import pyclesperanto_prototype as cle
from skimage.measure import regionprops
import python_calamine
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sklearn.metrics import roc_curve, auc
from matplotlib.widgets import Button
from mpl_interactions import panhandler, zoom_factory
import pickle
import torchvision.ops as ops
from matplotlib.lines import Line2D

'''
This code has the vizualization for the relevant cells and the bounding boxes
'''
from dotenv import load_dotenv

load_dotenv()
ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
dbx = dropbox.Dropbox(ACCESS_TOKEN)



#All the relevant info is in the
class InferenceWithPatch:
    def __init__(self, inform_files, granzyme_b_image_folder, processed_data_folder, model_path, patch_size, manual=True):
        self.inform_files = inform_files
        self.granzyme_b_image_folder = granzyme_b_image_folder
        self.processed_data_folder = processed_data_folder
        self.errors = []
        self.model_path = model_path
        self.patch_size = patch_size
        self.manual = manual

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.get_model(num_classes=2)  # 2 classes: background and Granzyme B
        #self.model.load_state_dict(torch.load(model_path))
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.to(self.device)


    def get_model(self, num_classes):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
        model.roi_heads.mask_predictor = None
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def list_files_in_folder(self, folder_path):
        files = []
        result = dbx.files_list_folder(folder_path)
        files.extend([entry.name for entry in result.entries if isinstance(entry, dropbox.files.FileMetadata)])
        while result.has_more:
            result = dbx.files_list_folder_continue(result.cursor)
            files.extend([entry.name for entry in result.entries if isinstance(entry, dropbox.files.FileMetadata)])
        return files

    def read_image_from_dropbox(self, dropbox_path):
        _, res = dbx.files_download(path=dropbox_path)
        file_bytes = io.BytesIO(res.content)
        image = skio.imread(file_bytes)
        return image

    def read_excel_from_dropbox(self, dropbox_path):
        _, res = dbx.files_download(path=dropbox_path)
        file_bytes = io.BytesIO(res.content)
        rows = iter(python_calamine.CalamineWorkbook.from_filelike(file_bytes).get_sheet_by_index(0).to_python())
        headers = list(map(str, next(rows)))
        data = [dict(zip(headers, row)) for row in rows]
        return pd.DataFrame(data)

    def relevant_rows(self, image_name, inform_excel):
        relevant_cells = pd.DataFrame()
        for index, row in inform_excel.iterrows():
            if image_name in row['Sample Name']:
                relevant_cells = relevant_cells._append(row)
        return relevant_cells

    def global_inform_values(self, inform_df):
        mean_nuclei = np.mean(inform_df['Nucleus Autofluorescence Mean (Normalized Counts, Total Weighting)'])
        std_nuclei = np.std(inform_df['Nucleus Autofluorescence Mean (Normalized Counts, Total Weighting)'])
        auto_98 = mean_nuclei + 2 * std_nuclei
        return auto_98

    def label_relevant_cells(self, relevant_rows):
        cd8_mem = 'Membrane CD8 (Opal 570) Mean (Normalized Counts, Total Weighting)'
        cd8_cyt = 'Cytoplasm CD8 (Opal 570) Mean (Normalized Counts, Total Weighting)'
        gb_mem = 'Membrane Granzyme B (Opal 620) Mean (Normalized Counts, Total Weighting)'
        gb_cyt = 'Cytoplasm Granzyme B (Opal 620) Mean (Normalized Counts, Total Weighting)'
        gb_ent = 'Entire Cell Granzyme B (Opal 620) Mean (Normalized Counts, Total Weighting)'
        cd8_ent = 'Entire Cell CD8 (Opal 570) Mean (Normalized Counts, Total Weighting)'

        relevant_cells = relevant_rows.copy()
        relevant_cells.dropna(subset=[cd8_mem, cd8_cyt, gb_mem, gb_cyt, gb_ent], inplace=True)

        relevant_cells['Label'] = 'None'

        relevant_cells[cd8_mem] = pd.to_numeric(relevant_cells[cd8_mem], errors='coerce')
        relevant_cells[cd8_cyt] = pd.to_numeric(relevant_cells[cd8_cyt], errors='coerce')
        relevant_cells[gb_mem] = pd.to_numeric(relevant_cells[gb_mem], errors='coerce')
        relevant_cells[gb_cyt] = pd.to_numeric(relevant_cells[gb_cyt], errors='coerce')
        relevant_cells[gb_ent] = pd.to_numeric(relevant_cells[gb_ent], errors='coerce')
        relevant_cells[cd8_ent] = pd.to_numeric(relevant_cells[cd8_ent], errors='coerce')

        #treshold_cd8 = np.median(relevant_cells[cd8_ent])
        print('Here')
        cd8_cells = relevant_cells[relevant_cells['Phenotype'] == 'CD8']
        treshold_cd8 = np.median(cd8_cells[cd8_ent])
        #print('Treshold:', treshold_cd8)
        #treshold_cd8 = np.percentile(cd8_cells[cd8_ent], 90) 
        #treshold_90_cd8 = np.percentile(treshold_cd8[cd8_ent], 90)
        treshold_cd8 = 50

        pos_phen = ['CD8', 'CD4', 'CD56']
        #row['Tissue Category'] == 'Tumor' 
        not_in_phen = 0
        not_gb = 0
        gbcy_gbm = 0
        gb_cd8 = 0
        print(max(relevant_cells[cd8_ent]), min(relevant_cells[cd8_ent]), treshold_cd8)
        # Add the labels to the relevant cells dataframe
        for index, row in relevant_cells.iterrows():
            if row['Phenotype'] in pos_phen:
                if row[gb_ent] > 0 and row[gb_cyt] > 0:
                    #if row[gb_cyt] >= row[gb_mem]:
                    if row['Phenotype'] == 'CD8':
                        # Check for potential interference
                        #
                        if row[cd8_ent] >= treshold_cd8:
                        
                            relevant_cells.at[index, 'Label'] = 'gb_cd8'  
                            gb_cd8 += 1
                    else:
                        relevant_cells.at[index, 'Label'] = 'gb'
                    #else:
                    #    relevant_cells.at[index, 'Label'] = 'gbcy_gbm'
                    #    gbcy_gbm += 1
                else:
                    relevant_cells.at[index, 'Label'] = 'not_gb'  # Directly classify as GrB+ for CD4/CD56
            else:
                relevant_cells.at[index, 'Label'] = 'not_phen'
                not_in_phen += 1

        print('Number of cells not in phen:', not_in_phen)
        print('Number of cells not gb:', not_gb)
        print('Number of cells gbvy_gbm:', gbcy_gbm)
        print('Number of cells gbct_cd8:', gb_cd8)

        positive_gb_cells = relevant_cells[relevant_cells['Label'] == 'gb']
        #not_phen_cells = relevant_cells[relevant_cells['Label'] == 'not_phen']
        #not_gb_cells = relevant_cells[relevant_cells['Label'] == 'not_gb']
        #gbcy_gbm_cells = relevant_cells[relevant_cells['Label'] == 'gbcy_gbm']
        #gbct_cd8_cells = relevant_cells[relevant_cells['Label'] == 'gbct_cd8']

        print('Number of positive Granzyme B cells:', len(positive_gb_cells))
        print('Number of  cells:', len(relevant_cells))
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


    def generate_bounding_boxes(self, patch, criteria_labels, pred_boxes, pred_scores,i_offset, j_offset):
        
        boxes = []
        print('Starting processing...')
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(patch)
        relevant_rows = criteria_labels.copy()

        margin = 0  # Large margin to accommodate the possible bounding boxes

        score_treshold = 0.2 # Has a low confidence threshold eg 0.5 to avoid false negatives and allow for manual elimination

        pred_boxes = np.array(pred_boxes)
        pred_scores = np.array(pred_scores)

        sorted_indices = np.argsort(pred_scores)[::-1]

        sorted_pred_boxes = pred_boxes[sorted_indices]
        sorted_pred_scores = pred_scores[sorted_indices]

        relevant_rows = relevant_rows.sort_values(by='Entire Cell Granzyme B (Opal 620) Mean (Normalized Counts, Total Weighting)', ascending=False)


        done_point = []

        # adjust relevant rows so we only look at the cells in the patch
        relevant_rows = relevant_rows.copy()
        x_max_patch = j_offset + patch.shape[1]
        y_max_patch = i_offset + patch.shape[0]
        x_min_patch = j_offset
        y_min_patch = i_offset
        relevant_rows = relevant_rows[(relevant_rows['Cell X Position'] >= x_min_patch) & (relevant_rows['Cell X Position'] <= x_max_patch)]
        relevant_rows = relevant_rows[(relevant_rows['Cell Y Position'] >= y_min_patch) & (relevant_rows['Cell Y Position'] <= y_max_patch)]

        
        for index, row in relevant_rows.iterrows():
            x0 = row['Cell X Position']
            y0 = row['Cell Y Position']
            #gb_cd8
            if row['Label'] == 'gb':
                ax.plot(x0 - j_offset, y0 - i_offset, 'o', color='green', markersize=4)
                #ax.text(x0 - j_offset, y0 - i_offset - 3, 'gb', color='black', fontsize=8, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
            elif row['Label'] == 'not_gb':
                ax.plot(x0 - j_offset, y0 - i_offset, 'o', color='blue', markersize=4)
               # ax.text(x0 - j_offset, y0 - i_offset - 3, 'no gb', color='black', fontsize=8, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
            elif row['Label'] == 'gb_cd8':
                ax.plot(x0 - j_offset, y0 - i_offset, 'o', color='purple', markersize=4)
                #ax.text(x0 - j_offset, y0 - i_offset - 3, 'gb cyto < cd8 cyto', color='black', fontsize=8, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
            elif row['Label'] == 'not_phen':
                ax.plot(x0 - j_offset, y0 - i_offset, 'o', color='red', markersize=4)
                #ax.text(x0 - j_offset, y0 - i_offset - 3, 'not phenotype', color='black', fontsize=8, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
            elif row['Label'] == 'gbcy_gbm':
                ax.plot(x0 - j_offset, y0 - i_offset, 'o', color='black', markersize=4)
                #ax.text(x0 - j_offset, y0 - i_offset - 3, 'gb cyto > gb mem', color='black', fontsize=8, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
           
        for pred_box, score in zip(sorted_pred_boxes, sorted_pred_scores):
            if score < score_treshold:
                continue
            x_min, y_min, x_max, y_max = pred_box
            rect = mpatches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor='blue', facecolor='none', linestyle='dashed'
            )
            ax.add_patch(rect)


            expanded_minc = max(0, x_min - margin) + j_offset
            expanded_maxc = x_max + margin + j_offset
            expanded_minr = max(0, y_min - margin) + i_offset
            expanded_maxr = y_max + margin + i_offset

            rel = relevant_rows[(relevant_rows['Label'] == 'gb')]

            for _, row in rel.iterrows():
                x_position = row['Cell X Position']
                y_position = row['Cell Y Position']

                if (x_position, y_position) in done_point:
                    continue

                if expanded_minc <= x_position <= expanded_maxc and expanded_minr <= y_position <= expanded_maxr:
                    #print("yess")
                    #ax.plot(x_position - j_offset, y_position - i_offset, 'o', color='purple', markersize=4)
                    #ph = row['Phenotype']
                    #ax.text(x_position - j_offset, y_position - i_offset - 3, ph, color='black', fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
                    done_point.append((x_position, y_position))


                    boxes.append({
                        'Cell ID': row['Cell ID'],
                        'X Position': int(x_position),
                        'Y Position': int(y_position),
                        'Bounding Box': [int(x_min + j_offset), int(y_min + i_offset), int(x_max+ j_offset), int(y_max + i_offset)],
                        'Granzyme B': row['Entire Cell Granzyme B (Opal 620) Mean (Normalized Counts, Total Weighting)'],
                        'Score': score
                    })
                    # Plots the patches 
                    rect = mpatches.Rectangle(
                        (x_min, y_min), x_max - x_min, y_max - y_min,
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.text(x_min, y_min - 10, f'{score:.2f}', color='black', fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
                    #break
        
        custom_lines = [
            Line2D([0], [0], color='green', marker='o', linestyle='None', markersize=6, label='Positive GB'),
            Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=6, label='No Granzyme B'),
            Line2D([0], [0], color='purple', marker='o', linestyle='None', markersize=6, label='CD8 leakage'),
            Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=6, label='Not relevant phenotype'),
            Line2D([0], [0], color='black', marker='o', linestyle='None', markersize=6, label='GB Cytoplasm < Membrane'),
            mpatches.Patch(edgecolor='blue', linestyle='dashed', fill=False, label='Model predicted Box'),
            mpatches.Patch(edgecolor='red', linestyle='solid', fill=False, label= 'Refined Box')
        ]
        # make the legend solid white, i.e., facecolor='white' and alpha=1
        #leg = ax.legend(handles=custom_lines, loc='upper right')
        #leg.get_frame().set_facecolor('white')
        #leg.get_frame().set_alpha(1)

        plt.axis('off')
        #plt.legend()
        plt.tight_layout()
        plt.show()
        return boxes


    def reconstruct_image(self, original_image, patches_with_boxes, patch_size):
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(original_image)
        for patch, patch_boxes, i, j in patches_with_boxes:
            for patch_box in patch_boxes:
                bb = patch_box["Bounding Box"]
                x_min, y_min, x_max, y_max = bb
                rect = mpatches.Rectangle(
                    #(x_min , y_min), (x_max-j) - (x_min-j), (y_max-i) - (y_min-i),
                    (x_min, y_min ), x_max - x_min, y_max - y_min,
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

        # if you return fig can you plot later

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
    

    def full_processing(self, image, patch_size, criteria_label):
        patches = self.create_patches_no_box(image, patch_size)
        patches = self.run_inference_on_patches(patches)
        patches_with_boxes = []

        for patch, pred_boxes, pred_scores, i, j in patches:
            #properties = self.process_image(patch)
            properties = None
            print('Generating bounding boxes...')
            boxes = self.generate_bounding_boxes(patch, criteria_label, pred_boxes, pred_scores, i, j)
            patches_with_boxes.append((patch, boxes, i, j))

        all_boxes = [box for _, boxes, _, _ in patches_with_boxes for box in boxes]
        print('Number of boxes:', len(all_boxes))
        print('Type of all_boxes:', type(all_boxes))

        if self.manual:
            valid_boxes = self.manual_elimination(image, all_boxes)
            exlude_boxes = [box for box in all_boxes if box not in valid_boxes]
            return valid_boxes

        new_image = self.reconstruct_image(image, patches_with_boxes, patch_size)
        self.plot_image_with_boxes(new_image, all_boxes, title="Reconstructed Image with Bounding Boxes")
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
        #plt.imshow(segmented_np)

        properties = measure.regionprops(segmented_np, intensity_image=H[:, :, 0])
        # print number of cells

   

        #ax.axis('off')
        #plt.show()
        return properties

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
    
    def plot_image_with_inform(self, image, inform_excel):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)

        #pos_phen = ['CD8', 'CD4', 'CD56']
        for index, row in inform_excel.iterrows():
            x_position = row['Cell X Position']
            y_position = row['Cell Y Position']
            ax.plot(x_position, y_position, 'o', color='red', markersize=2)

        ax.plot(0, 0, 'o', color='green', markersize=6)

        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def read_txt_from_dropbox(self, dropbox_path, delimiter='\t'):
        """
        Reads a .txt file from Dropbox and loads it as a pandas DataFrame.
        
        Args:
        dropbox_path (str): The path to the .txt file in Dropbox.
        delimiter (str): The delimiter used in the .txt file (default is tab '\t').
        
        Returns:
        pd.DataFrame: The loaded data as a DataFrame.
        """
        _, res = dbx.files_download(path=dropbox_path)
        file_bytes = io.BytesIO(res.content)
        # Assume the file is structured as a CSV or TSV
        return pd.read_csv(file_bytes, delimiter=delimiter)


    def run(self, example):
        inform_files = self.list_files_in_folder(self.inform_files)
        image_files = self.list_files_in_folder(self.granzyme_b_image_folder)
        
        for inform_file in inform_files:
            if inform_file.endswith('.txt'):
                inform_path = self.inform_files + inform_file
                inform_excel = self.read_txt_from_dropbox(inform_path)
                name_sample = inform_file.split(' Tumor merge_cell_seg_data.txt')[0]
                print('Starting comparison')
                if name_sample != example:
                    continue

                print(f'Working on sample: {name_sample}')
                #phenotypes = inform_excel['Phenotype'].unique()
                #print(f'Phenotypes in sample: {phenotypes}')
                #tissue_types = inform_excel['Tissue Category'].unique()
                #print(f'Tissue types in sample: {tissue_types}')
                relevant_images = [f for f in image_files if name_sample in f and 'Granzyme' in f]
                print(f'Number of images in sample: {len(relevant_images)}')
                #max_x_pos = inform_excel['Cell X Position'].max()
                #max_y_pos = inform_excel['Cell Y Position'].max()
                #print(f'Max X Position: {max_x_pos}')
                #print(f'Max Y Position: {max_y_pos}')
                #min_x_pos = inform_excel['Cell X Position'].min()
                #min_y_pos = inform_excel['Cell Y Position'].min()
                #print(f'Min X Position: {min_x_pos}')

                for image_file in relevant_images:
                    image_name = image_file.split('_Granzyme')[0]
                    try:
                        image_path = self.granzyme_b_image_folder + image_file
                        print(f'Processing image: {image_name}')
                        relevant_rows = self.relevant_rows(image_name, inform_excel)
                        criteria_labels= self.label_relevant_cells(relevant_rows)
                        gb_image = self.read_image_from_dropbox(image_path)
                        print('image shape:', gb_image.shape)
                        #self.plot_image_with_inform(gb_image, relevant_rows)
                        all_boxes = self.full_processing(gb_image, self.patch_size, criteria_labels)
                    except Exception as e:
                        print(f'Error processing {image_file}: {e}')
 

                print(f'Done processing sample: {name_sample}')



# Example usage:
inform_files = '/more CD8 leakage/Label/'
granzyme_b_image_folder = '/more CD8 leakage/'
processed_data_folder = 'data/processed_thresholds'
model_path =  'data/saved_models/10fine_40pre.pth'
#'data/saved_models/40_extra_relax_basic.pth'
#'data/saved_models/10fine_40pre.pth'
#data/saved_models/20e_trans_with_relax.pth
#data/saved_models/sextra_impie_then_manual.pth copy
patch_size = 256
example_sample = 'Opal 139_3'  # Example

inference = InferenceWithPatch(inform_files, granzyme_b_image_folder, processed_data_folder, model_path, patch_size)
inference.run(example_sample)
