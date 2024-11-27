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
import logging
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from scipy.stats import pearsonr


from dotenv import load_dotenv

load_dotenv()
ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
dbx = dropbox.Dropbox(ACCESS_TOKEN)


#All the relevant info is in the
class inFormModelEval:
    def __init__(self, inform_files, granzyme_b_image_folder, model_path, patch_size, manual= False):
        self.inform_files = inform_files
        self.granzyme_b_image_folder = granzyme_b_image_folder
        self.errors = []
        self.model_path = model_path
        self.patch_size = patch_size
        self.manual = manual
        self.gbcy_gbm_fp = 0
        self.gb_cd8_fp = 0
        self.not_gb_fp = 0
        self.not_phen_fp = 0
        self.true_positives = 0
        self.true_negative = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.non_matched_criteria = 0
        self.scores = []
        self.non_match_val = []
        self.truegb_val = []
        self.score_gb = []

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
        print('in label relevant cells')
        #aut_nuclei = 'Nucleus Autofluorescence Mean (Normalized Counts, Total Weighting)'
        cd8_mem = 'Membrane CD8 (Opal 570) Mean (Normalized Counts, Total Weighting)'
        cd8_cyt = 'Cytoplasm CD8 (Opal 570) Mean (Normalized Counts, Total Weighting)'
        gb_mem = 'Membrane Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)'
        gb_cyt = 'Cytoplasm Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)'
        gb_ent = 'Entire Cell Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)'
        cd8_ent = 'Entire Cell CD8 (Opal 570) Mean (Normalized Counts, Total Weighting)'


        relevant_cells = relevant_rows.copy()
        relevant_cells['Label'] = 'None'
        #relevant_cells[aut_nuclei] = pd.to_numeric(relevant_cells[aut_nuclei], errors='coerce')
        relevant_cells[cd8_mem] = pd.to_numeric(relevant_cells[cd8_mem], errors='coerce')
        relevant_cells[cd8_cyt] = pd.to_numeric(relevant_cells[cd8_cyt], errors='coerce')
        relevant_cells[gb_mem] = pd.to_numeric(relevant_cells[gb_mem], errors='coerce')
        relevant_cells[gb_cyt] = pd.to_numeric(relevant_cells[gb_cyt], errors='coerce')
        relevant_cells[gb_ent] = pd.to_numeric(relevant_cells[gb_ent], errors='coerce')
        relevant_cells[cd8_ent] = pd.to_numeric(relevant_cells[cd8_ent], errors='coerce')
        pos_phen = ['CD8', 'CD4', 'CD56']
        # 50% cutoff for CD8 signal
        print('Here')
        cd8_cells = relevant_cells[relevant_cells['Phenotype'] == 'CD8']
        #treshold_cd8 = np.median(cd8_cells[cd8_ent])
        #print('Treshold:', treshold_cd8)
        #tres_cd8_ent = np.median(relevant_rows[cd8_ent])
        #print('tres_cd8_ent:', tres_cd8_ent)
        cd8_cells = relevant_cells[relevant_cells['Phenotype'] == 'CD8']
        #treshold_cd8 = np.median(cd8_cells[cd8_ent])
        #print('Treshold:', treshold_cd8)
        treshold_cd8 = np.percentile(cd8_cells[cd8_ent], 90) 
        


        for index, row in relevant_cells.iterrows():
            if row['Phenotype'] in pos_phen:
                if row[gb_ent] > 0 and row[gb_cyt] > 0:
                    if row['Phenotype'] == 'CD8':
                        # Check for potential interference
                        if row[cd8_ent] >= treshold_cd8:
                            relevant_cells.at[index, 'Label'] = 'gb_cd8'  
                        else:
                            relevant_cells.at[index, 'Label'] = 'gb'  
                    else:
                        relevant_cells.at[index, 'Label'] = 'gb'  # Directly classify as GrB+ for CD4/CD56
                else:
                    relevant_cells.at[index, 'Label'] = 'not_gb'
            else:
                relevant_cells.at[index, 'Label'] = 'not_phen'

        positive_gb_cells = relevant_cells[relevant_cells['Label'] == 'gb']
        return positive_gb_cells, relevant_cells

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
        #fig, ax = plt.subplots(figsize=(10, 10))
        #ax.imshow(patch)
        relevant_rows = criteria_labels.copy()

        margin = 0  # Large margin to accommodate the possible bounding boxes

        score_treshold = 0 # Has a low confidence threshold eg 0.5 to avoid false negatives and allow for manual elimination

        pred_boxes = np.array(pred_boxes)
        pred_scores = np.array(pred_scores)

        sorted_indices = np.argsort(pred_scores)[::-1]

        sorted_pred_boxes = pred_boxes[sorted_indices]
        sorted_pred_scores = pred_scores[sorted_indices]

        done_point = []

        # adjust relevant rows so we only look at the cells in the patch
        #print('Here')
        relevant_rows['Cell X Position'] = pd.to_numeric(relevant_rows['Cell X Position'])
        relevant_rows['Cell Y Position'] = pd.to_numeric(relevant_rows['Cell Y Position'])
        x_max_patch = j_offset + patch.shape[1]
        y_max_patch = i_offset + patch.shape[0]
        x_min_patch = j_offset
        y_min_patch = i_offset
        relevant_rows = relevant_rows[(relevant_rows['Cell X Position'] >= x_min_patch) & (relevant_rows['Cell X Position'] <= x_max_patch)]
        relevant_rows = relevant_rows[(relevant_rows['Cell Y Position'] >= y_min_patch) & (relevant_rows['Cell Y Position'] <= y_max_patch)]

        relevant_rows['Entire Cell Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)'] = pd.to_numeric(relevant_rows['Entire Cell Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)']).dropna()
        #sort the relevant rows by the granzyme b value

        relevant_rows = relevant_rows.sort_values(by='Entire Cell Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)', ascending=False)

        done = set()

        gb_cell = []

        
        for pred_box, score in zip(sorted_pred_boxes, sorted_pred_scores):
            if score < score_treshold:
                continue
            x_min, y_min, x_max, y_max = pred_box
            rect = mpatches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor='blue', facecolor='none', linestyle='dashed'
            )
            #ax.add_patch(rect)
            self.scores.append(score)


            expanded_minc = max(0, x_min - margin) + j_offset
            expanded_maxc = x_max + margin + j_offset
            expanded_minr = max(0, y_min - margin) + i_offset
            expanded_maxr = y_max + margin + i_offset

            true = []
            sc = []
            count = 0
            mult = 0
            fp = 0

            for _, row in relevant_rows.iterrows():
                x_position = row['Cell X Position']
                y_position = row['Cell Y Position']


                if expanded_minc <= x_position <= expanded_maxc and expanded_minr <= y_position <= expanded_maxr:
                    #print("yess")
                    done_point.append((x_position, y_position))
                    done.add(row['Cell ID'])

                    #ax.plot(x_position - j_offset, y_position - i_offset, 'o', color='purple', markersize=4)
                    #ph = row['Phenotype']
                    #ax.text(x_position - j_offset, y_position - i_offset - 3, ph, color='black', fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
                    if row['Label'] == 'gb':
                        #self.true_positives += 1
                        #self.truegb_val.append(row['Entire Cell Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)'])
                        sc.append(score)
                        #gb_cell.append(row['Cell ID'])
                        true.append(row['Entire Cell Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)'])
                        count += 1
                        mult += 1
                    else:
                        mult += 1
                        fp += 1
                        if row['Label'] == 'gbcy_gbm':
                            self.gbcy_gbm_fp += 1
                        elif row['Label'] == 'gb_cd8':
                            self.gb_cd8_fp += 1
                        elif row['Label'] == 'not_gb':
                            self.not_gb_fp += 1
                        elif row['Label'] == 'not_phen':
                            self.not_phen_fp += 1

                    boxes.append({
                        'Cell ID': row['Cell ID'],
                        'X Position': int(x_position),
                        'Y Position': int(y_position),
                        'Bounding Box': [int(x_min + j_offset), int(y_min + i_offset), int(x_max+ j_offset), int(y_max + i_offset)],
                        'Granzyme B': row['Entire Cell Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)'],
                        'Score': score
                    })
                    # Plots the patches 
                    rect = mpatches.Rectangle(
                        (x_min, y_min), x_max - x_min, y_max - y_min,
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    #ax.add_patch(rect)
                    #ax.text(x_min, y_min - 10, f'{score:.2f}', color='black', fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
            if mult == 1:
                if count == 1:
                    self.true_positives += count
                    self.score_gb.extend(sc)
                    self.truegb_val.extend(true)
                else:
                    self.false_positives += 1
            elif mult > 1:
                if fp > 0:
                    self.false_positives += 1

        for _, row in relevant_rows.iterrows():
            if row['Cell ID'] not in done:
                if row['Label'] == 'gb':
                    self.non_matched_criteria += 1
                    self.non_match_val.append(row['Entire Cell Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)'])
                if row['Label'] != 'gb':
                    self.true_negative += 1



        #self.true_negative = len(relevant_rows) - self.true_positives - self.false_positives 
                

        # make the legend solid white, i.e., facecolor='white' and alpha=1
        #leg = ax.legend(handles=custom_lines, loc='upper right')
        #leg.get_frame().set_facecolor('white')
        #leg.get_frame().set_alpha(1)

        #plt.axis('off')
        #plt.legend()
        #plt.tight_layout()
        #plt.show()
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
    

    def full_processing(self, image, patch_size, criteria_labels, image_name):
        patches = self.create_patches_no_box(image, patch_size)
        patches = self.run_inference_on_patches(patches)
        patches_with_boxes = []

        model_pred_boxes = []
        print(len(patches))


        for patch, pred_boxes, pred_scores, i, j in patches:
            #properties = self.process_image(patch)
            model_pred_boxes.append(pred_boxes)
            #print(pred_boxes)
            print('Generating bounding boxes...')
            boxes = self.generate_bounding_boxes(patch,criteria_labels, pred_boxes, pred_scores, i, j)
            patches_with_boxes.append((patch, boxes, i, j))

        all_boxes = [box for _, boxes, _, _ in patches_with_boxes for box in boxes]
        print('Number of boxes:', len(all_boxes))
        print('Type of all_boxes:', type(all_boxes))
        

        return all_boxes, model_pred_boxes

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

        return properties

    def save_processed_data(self, sample_name, image_name, data):
        sample_folder = os.path.join(self.processed_data_folder, sample_name)
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
        file_path = os.path.join(sample_folder, f'{image_name}.pkl')
        print(f'Saving processed data to {file_path}')
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)


    def load_processed_data(self, sample_name, image_name):
        file_path = os.path.join(self.processed_data_folder, sample_name, f'{image_name}.pkl')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        return None
    
    def evaluate_and_visualize(self):
        print('Model evaluation results:')
        
        # Use class attributes
        TP = self.true_positives
        FP = self.false_positives
        FN = self.false_negatives  # False Negatives (traditional)
        NM = self.non_matched_criteria  # Special case: non-matched criteria
        TN = self.true_negative 

        print(TP, FP, FN, NM, TN)

        # Evaluate Non-Matched Criteria and Adjust False Negatives
        if NM > 0 and TP > 0:

            self.score_gb = np.array(self.score_gb)
            self.truegb_val = np.array(self.truegb_val)

            #lower_10_percentile = np.percentile(self.truegb_val, 10)
            gb_mdeian = np.median(self.truegb_val)
            upper_90_percentile = np.percentile(self.truegb_val, 90)
            lower_10_percentile = np.percentile(self.truegb_val, 10)


            gb_50 = []
            score_50 = []

            #
            non_matched_values = self.non_match_val
            #print(non_matched_values, max(non_matched_values), min(non_matched_values))
            true_positive_values = self.truegb_val
            
            # Calculate means
            mean_nm = np.mean(non_matched_values)
            mean_tp = np.mean(true_positive_values)
            print(f"Mean of Non-matched Criteria: {mean_nm:.4f}")
            print(f"Mean of True Positives: {mean_tp:.4f}")


            for i in range(len(self.score_gb)):
                if self.score_gb[i] >= 0.5 and self.truegb_val[i] >= lower_10_percentile:
                    gb_50.append(self.truegb_val[i])
                    score_50.append(self.score_gb[i])

            #if len(gb_50) == 0:
                

            print(f"Number of True Positives with Score > 0.5 (and in the 90% of the true GB distribution): {len(gb_50)}")
            
            # Identify non-matched values that should be reclassified as false negatives
            if len(gb_50) > 0:
                lowest_tp_value = min(gb_50)
                reclassified_fn = [val for val in non_matched_values if val >= lowest_tp_value]
                self.false_negatives += len(reclassified_fn)
                FN += len(reclassified_fn)  
                TN += (NM - len(reclassified_fn))  
                print(f"Reclassified {len(reclassified_fn)} non-matched criteria as false negatives.")
            else:
                print("No non-matched criteria to reclassify.")
                lowest_tp_value = min(true_positive_values)

            # Plot distributions
            plt.figure(figsize=(10, 5))
            
            # Separate distributions
            plt.subplot(1, 2, 1)
            sns.histplot(true_positive_values, kde=True, color='green', label='True Positives')
            plt.axvline(mean_tp, color='green', linestyle='--', label=f'Mean TP: {mean_tp:.4f}')
            plt.title('Distribution of True Positives')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()

            plt.subplot(1, 2, 2)
            sns.histplot(non_matched_values, kde=True, color='orange', label='Non-Matched Criteria')
            plt.axvline(mean_nm, color='orange', linestyle='--', label=f'Mean NM: {mean_nm:.4f}')
            plt.title('Distribution of Non-Matched Criteria')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()

            plt.tight_layout()
            plt.show()

            # Overlapping distributions
            plt.figure(figsize=(10, 5))
            sns.histplot(true_positive_values, kde=True, color='green', label='True Positives', alpha=0.5)
            sns.histplot(non_matched_values, kde=True, color='orange', label='Non-Matched Criteria', alpha=0.5)
            plt.axvline(mean_tp, color='green', linestyle='--', label=f'Mean TP: {mean_tp:.4f}')
            plt.axvline(mean_nm, color='orange', linestyle='--', label=f'Mean NM: {mean_nm:.4f}')
            plt.axvline(lowest_tp_value, color='red', linestyle='--', label='Lowest 0.5 TP Value')
            plt.title('Overlapping Distributions')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()
            


            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            sns.boxplot(x=true_positive_values, color='green')
            plt.title('Boxplot of True Positives')
            plt.xlabel('Value')

            plt.subplot(1, 2, 2)
            sns.boxplot(x=non_matched_values, color='orange')
            plt.title('Boxplot of Non-Matched Criteria')
            plt.xlabel('Value')

            plt.tight_layout()
            plt.show()

            log_mean_tp = np.log10(np.mean(true_positive_values))
            log_mean_nm = np.log10(np.mean(non_matched_values))

            # Step 2: Plot the distributions with log scale
            plt.figure(figsize=(10, 5))

            # Plot for True Positives
            plt.subplot(1, 2, 1)
            sns.histplot(true_positive_values, kde=True, color='green', log_scale=True, label='True Positives')
            plt.axvline(log_mean_tp, color='green', linestyle='--', label=f'Log Mean TP: {log_mean_tp:.4f}')
            plt.title('Distribution of True Positives (Log Scale)')
            plt.xlabel('Log(Value)')
            plt.ylabel('Frequency')
            plt.legend()

            # Plot for Non-Matched Criteria
            plt.subplot(1, 2, 2)
            sns.histplot(non_matched_values, kde=True, color='orange', log_scale=True, label='Non-Matched Criteria')
            plt.axvline(log_mean_nm, color='orange', linestyle='--', label=f'Log Mean NM: {log_mean_nm:.4f}')
            plt.title('Distribution of Non-Matched Criteria (Log Scale)')
            plt.xlabel('Log(Value)')
            plt.ylabel('Frequency')
            plt.legend()

            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 5))
            sns.histplot(true_positive_values, kde=True, color='green', log_scale=True, label='True Positives')
            sns.histplot(non_matched_values, kde=True, color='orange', log_scale=True, label='Non-Matched Criteria')
            plt.axvline(log_mean_tp, color='green', linestyle='--', label=f'Log Mean TP: {log_mean_tp:.4f}')
            plt.axvline(log_mean_nm, color='orange', linestyle='--', label=f'Log Mean NM: {log_mean_nm:.4f}')
            #plt.axvline(np.log10(lowest_tp_value), color='red', linestyle='--', label='Lowest 0.5 TP Value')
            plt.title('Overlapping Distributions (Log Scale)')
            plt.xlabel('Log(Value)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()

        else:
            print("No non-matched criteria to analyze.")

        # Precision, Recall, and F1 Score
        if TP + FP > 0:
            precision = TP / (TP + FP)
        else:
            precision = 0.0

        if TP + FN > 0:
            recall = TP / (TP + FN)
        else:
            recall = 0.0

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Confusion Matrix Visualization (excluding NM from the main matrix)
        cm = np.array([
            [TP, FN],
            [FP, TN]
        ])


        #tp, fn, fp, tn

        sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Predicted Positive', 'Predicted Negative'], yticklabels=['Actual Positive', 'Actual Negative'])
        plt.title('Confusion Matrix')
        plt.show()

        total = TP + FN + FP + TN
        tp_p = (TP / total) * 100
        fn_p = (FN / total) * 100
        fp_p = (FP / total) * 100
        tn_p = (TN / total) * 100

        cm_percentage = np.array([
            [tp_p, fn_p],
            [fp_p, tn_p]
        ])

        #cm_percentage = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='coolwarm', xticklabels=['Predicted Positive', 'Predicted Negative'], yticklabels=['Actual Positive', 'Actual Negative'])
        plt.title('Confusion Matrix (Percentage)')
        plt.show()

        # Plot distribution of predicted scores and mean line
        if len(self.scores) > 0:
            plt.hist(self.scores, bins=20, color='skyblue', edgecolor='black', linewidth=1.2)
            #plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold')
            plt.title('Distribution of Predicted Scores')
            plt.xlabel('Predicted Score')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()
        else:
            print("No predicted scores available to plot.")

        if TP > 0 and len(self.truegb_val) > 1 and len(self.score_gb) > 1:
            # Calculate Pearson correlation
            corr, p_value = pearsonr(self.truegb_val, self.score_gb)

            # Create the scatter plot
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=self.score_gb, y=self.truegb_val, color='blue', alpha=0.5)

            # Fit a linear line to the scatter plot
            m, b = np.polyfit(self.score_gb, self.truegb_val, 1)
            plt.plot(self.score_gb, m * np.array(self.score_gb) + b, color='red', label=f'Fit line (slope={m:.2f}) Pearsons Correlation: r = {corr:.3f} (P = {p_value:.4g})')

            # Annotate the plot with the Pearson correlation coefficient
            #plt.text(0.05, max(self.truegb_val) * 0.9, f"Pearson's Correlation: r = {corr:.3f} (P = {p_value:.4g})", 
            #        fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

            # Label the axes and title
            plt.xlabel('Predicted Probability (Scores)')
            plt.ylabel('Granzyme B Values')
            plt.title('Correlation between True Granzyme B Values and Predicted Probability')
            plt.legend()
            plt.show()

            corr_t, p_value_t = pearsonr(self.truegb_val, self.score_gb)

            #correlation 
            if len(gb_50) > 0:
                corr_50, p_value_50 = pearsonr(gb_50, score_50)

                # Create the scatter plot
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=score_50, y=gb_50, color='purple', alpha=0.5)

                # Fit a linear line to the scatter plot
                m, b = np.polyfit(score_50, gb_50, 1)
                plt.plot(score_50, m * np.array(score_50) + b, color='red', label=f'Fit line (slope={m:.2f}) Pearsons Correlation: r = {corr_50:.3f} (P = {p_value_50:.4g})')

                plt.xlabel('Predicted Probability (Scores that are > 0.5)')
                plt.ylabel('Granzyme B Values')
                plt.title('Correlation between True Granzyme B Values (90% percentile) and Predicted Probability (Scores > 0.5)')
                plt.legend()
                plt.show()

        else:
            print("Not enough data to calculate Pearson correlation.")

        results = {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Confusion Matrix': cm,
            'Scores': self.scores,
            'GB_50': gb_50,
            'Score_50': score_50,
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN,
            'NM': NM,
        }

        return precision, recall, f1, cm, self.scores, gb_50, score_50, TP, FP, FN, TN, NM



    def run_model_val(self):
        inform_files_main = self.list_files_in_folder(self.inform_files)
        image_files = self.list_files_in_folder(self.granzyme_b_image_folder)
        errors = os.listdir('/Users/rebeca/Documents/GitHub/SHY_deep_learning/data/refine_model_processed_thresholds/errors')
        errors = [f.split('_error.json')[0] for f in errors]
        #print(inform_files_main)

        #radom 5 fil
        # repeat it 3 times to get an average

        for i in range(1):

            #inform_files = inform_files_main[6:7]
            inform_files = inform_files_main[0:1]

            valid_cells = []
            predicted_cells = []
            excel_files = []


            for inform_file in inform_files:
                if inform_file.endswith('.xlsx'):
                    inform_path = self.inform_files + inform_file
                    inform_excel = self.read_excel_from_dropbox(inform_path)
                    name_sample = inform_file.split('.xlsx')[0]


                    print(f'Working on sample: {name_sample}')
                    relevant_images = [f for f in image_files if name_sample in f and 'Granzyme' in f]
                    #relevant_images = np.random.choice(relevant_images, 5)
                    relevant_images = relevant_images[0:1]
                    #relevant_images = relevant_images[0:5]


                    print(f'Number of images in sample: {len(relevant_images)}')

                    for image_file in relevant_images:
                        image_name = image_file.split('_Granzyme')[0]

                        if f'{image_name}' in errors:
                            print(f'Skipping image {image_name} due to previous error')
                            continue

                        try:
                            image_path = self.granzyme_b_image_folder + image_file
                            print(f'Processing image: {image_name}')
                            relevant_rows = self.relevant_rows(image_name, inform_excel)
                            criteria_labels, relevant_cells = self.label_relevant_cells(relevant_rows)
                            gb_image = self.read_image_from_dropbox(image_path)
                            valid_boxes, all_boxes  = self.full_processing(gb_image, self.patch_size, relevant_cells, image_name)
                            #valid_cells.extend(valid_boxes)
                            #print('Extended valid cells')
                            #predicted_cells.extend(all_boxes)
                            #print('Extended predicted cells')
                            #excel_files.extend(relevant_cells)
                            #print('Extended excel files')
                            
                        except Exception as e:
                            self.errors.append({'image': image_file, 'error': str(e)})
                            print(f'Error processing {image_file}: {e}')
                            #end the code here
                            #j
                            # son_path = f'data/refine_model_processed_thresholds/errors/{image_name}_error.json'
                            #with open(json_path, 'w') as json_file:
                            #    json.dump(image_name, json_file)
                            continue

        print(f'Starting model evaluation')
        self.evaluate_and_visualize()

def compare_models(model_paths, inform_files, granzyme_b_image_folder, patch_size):
    for model_path in model_paths:
        inference = inFormModelEval(inform_files, granzyme_b_image_folder, model_path, patch_size)
        inference.run_model_val()

# Example usage:
inform_files = '/Rebeca&Laura/inform_in_excel/'
#granzyme_b_image_folder = '/UEC, CD8 and GranzymeB/'
granzyme_b_image_folder = '/UEC, CD8 and GranzymeB/test/'
model_path = 'data/saved_models/10e_trans_with_relax.pth'
#'data/saved_models/10fine_40pre.pth'
patch_size = 256

models_to_analyze = ['data/saved_models/10e_trans_with_relax.pth', 'data/saved_models/20e_trans_with_relax.pth', 'data/saved_models/original_more_epoch_extra_relax_basic copy.pth', 'data/saved_models/with_trans_longish.pth']

inference = inFormModelEval(inform_files, granzyme_b_image_folder, model_path, patch_size)
inference.run_model_val()

