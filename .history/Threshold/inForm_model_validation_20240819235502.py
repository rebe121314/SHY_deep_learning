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

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.get_model(num_classes=2)  # 2 classes: background and Granzyme B
        self.model.load_state_dict(torch.load(model_path))
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
        #aut_nuclei = 'Nucleus Autofluorescence Mean (Normalized Counts, Total Weighting)'
        cd8_mem = 'Membrane CD8 (Opal 570) Mean (Normalized Counts, Total Weighting)'
        cd8_cyt = 'Cytoplasm CD8 (Opal 570) Mean (Normalized Counts, Total Weighting)'
        gb_mem = 'Membrane Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)'
        gb_cyt = 'Cytoplasm Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)'
        gb_ent = 'Entire Cell Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)'

        relevant_cells = relevant_rows.copy()
        relevant_cells['Label'] = 'None'

        #relevant_cells[aut_nuclei] = pd.to_numeric(relevant_cells[aut_nuclei], errors='coerce')
        relevant_cells[cd8_mem] = pd.to_numeric(relevant_cells[cd8_mem], errors='coerce')
        relevant_cells[cd8_cyt] = pd.to_numeric(relevant_cells[cd8_cyt], errors='coerce')
        relevant_cells[gb_mem] = pd.to_numeric(relevant_cells[gb_mem], errors='coerce')
        relevant_cells[gb_cyt] = pd.to_numeric(relevant_cells[gb_cyt], errors='coerce')
        relevant_cells[gb_ent] = pd.to_numeric(relevant_cells[gb_ent], errors='coerce')
        pos_phen = ['CD8', 'CD4', 'CD56']

        for index, row in relevant_cells.iterrows():
            if row['Phenotype'] in pos_phen:
                if row[gb_ent] > 0 and row[gb_cyt] > 0:
                    #if row[aut_nuclei] > auto_98:
                    if row[gb_cyt] >= row[gb_mem]:
                        if row[gb_cyt] >= row[cd8_cyt]:
                            relevant_cells.at[index, 'Label'] = 'gb'
                        else:
                            #gbct_cd8 += 1
                            relevant_cells.at[index, 'Label'] = 'gbct_cd8'
                    else:
                        #gbcy_gbm += 1
                        relevant_cells.at[index, 'Label'] = 'gbcy_gbm'
                else:
                    #not_gb += 1
                    relevant_cells.at[index, 'Label'] = 'not_gb'
            else:
                #not_in_phen += 1
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

    def generate_bounding_boxes(self, patch, pred_boxes, pred_scores, criteria_labels, i_offset, j_offset):
        boxes = []
        #fig, ax = plt.subplots(figsize=(10, 10))
        #ax.imshow(patch)

        margin = 0  # Large margin to accommodate the possible bounding boxes

        score_treshold = 0 # Has a low confidence threshold eg 0.5 to avoid false negatives and allow for manual elimination

        pred_boxes = np.array(pred_boxes)
        pred_scores = np.array(pred_scores)

        sorted_indices = np.argsort(pred_scores)[::-1]

        sorted_pred_boxes = pred_boxes[sorted_indices]
        sorted_pred_scores = pred_scores[sorted_indices]
        done_point = []

        for pred_box, score in zip(sorted_pred_boxes, sorted_pred_scores):
            if score < score_treshold:
                continue
            x_min, y_min, x_max, y_max = pred_box
            rect = mpatches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor='blue', facecolor='none', linestyle='dashed'
            )
            #ax.add_patch(rect)

            expanded_minc = max(0, x_min - margin) + j_offset
            expanded_maxc = x_max + margin + j_offset
            expanded_minr = max(0, y_min - margin) + i_offset
            expanded_maxr = y_max + margin + i_offset

            for _, row in criteria_labels.iterrows():
                x_position = row['Cell X Position']
                y_position = row['Cell Y Position']

                if (x_position, y_position) in done_point:
                    continue
                
                if expanded_minc <= x_position <= expanded_maxc and expanded_minr <= y_position <= expanded_maxr:
                    #print("yess")
                    #ax.plot(x_position - j_offset, y_position - i_offset, 'o', color='black', markersize=4)
                    done_point.append((x_position, y_position))

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
                    break


        #plt.axis('off')
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

        criteria_labels = criteria_labels[criteria_labels['Label'] == 'gb']

        for patch, pred_boxes, pred_scores, i, j in patches:
            #properties = self.process_image(patch)
            model_pred_boxes.append(pred_boxes)
            #print(pred_boxes)
            print('Generating bounding boxes...')
            boxes = self.generate_bounding_boxes(patch, pred_boxes, pred_scores, criteria_labels, i, j)
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
    
    def model_eval(self, criteria_labels, valid_boxes, predicted_boxes):
        print('Evaluating model...')
        gbcy_gbm_fp = 0
        gbct_cd8_fp = 0
        not_gb_fp = 0
        not_phen_fp = 0
        true_positives = 0
        false_positives = 0
        non_matched_criteria = 0

        for pred_box in predicted_boxes:
            print(pred_box)
            x_min, y_min, x_max, y_max = pred_box
            matched = False
            
            for criteria_df in criteria_labels:
                print('Analyzing {}'.format(criteria_df))
                for _, row in criteria_df.iterrows():
                    x_position = row['Cell X Position']
                    y_position = row['Cell Y Position']
                    
                    if x_min <= x_position <= x_max and y_min <= y_position <= y_max:
                        matched = True
                        if row['Label'] == 'gb':
                            true_positives += 1
                        else:
                            false_positives += 1
                            if row['Label'] == 'gbcy_gbm':
                                gbcy_gbm_fp += 1
                            elif row['Label'] == 'gbct_cd8':
                                gbct_cd8_fp += 1
                            elif row['Label'] == 'not_gb':
                                not_gb_fp += 1
                            elif row['Label'] == 'not_phen':
                                not_phen_fp += 1
                        break
                if matched:
                    break

            if not matched and row['Label'] == 'gb':
                    non_matched_criteria += 1
            else:
                false_positives += 1
 

        #plot all_scores
        all_scores = [box['Score'] for box in predicted_boxes]
        mean_score = np.mean(all_scores)
        plt.hist(all_scores, bins=20)
        plt.title('Model Prediction Scores')
        plt.axvline(mean_score, color='red', linestyle='dashed', linewidth=1, label=f'Mean Score: {mean_score:.2f}')
        plt.legend()
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.show()

        # Return results for further analysis
        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'non_matched_criteria': non_matched_criteria,
            'gbcy_gbm_fp': gbcy_gbm_fp,
            'gbct_cd8_fp': gbct_cd8_fp,
            'not_gb_fp': not_gb_fp,
            'not_phen_fp': not_phen_fp
        }
    
    def evaluate_and_visualize(results):
        print('Model evaluation results:')
        TP = results['true_positives']
        FP = results['false_positives']
        NM = results['non_matched_criteria']
        
        precision = precision_score([1] * TP + [0] * FP, [1] * TP + [1] * FP)
        recall = recall_score([1] * TP + [1] * NM, [1] * TP + [0] * NM)
        f1 = f1_score([1] * TP + [1] * NM, [1] * TP + [0] * NM)
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Confusion Matrix Visualization
        cm = confusion_matrix([1] * TP + [0] * FP, [1] * TP + [1] * FP, labels=[1, 0])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['TP', 'FP'], yticklabels=['Actual', 'Predicted'])
        plt.title('Confusion Matrix')
        plt.show()


    def run_model_val(self):
        inform_files = self.list_files_in_folder(self.inform_files)
        image_files = self.list_files_in_folder(self.granzyme_b_image_folder)
        errors = os.listdir('/Users/rebeca/Documents/GitHub/SHY_deep_learning/data/refine_model_processed_thresholds/errors')
        errors = [f.split('_error.json')[0] for f in errors]

        #radom 5 fil

        inform_files = np.random.choice(inform_files, 1)
        #inform_files = inform_files[0]

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
                relevant_images = np.random.choice(relevant_images, 1)
                #relevant_images = relevant_images[0]

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
                        valid_cells.extend(valid_boxes)
                        print('Extended valid cells')
                        predicted_cells.extend(all_boxes)
                        print('Extended predicted cells')
                        excel_files.extend(relevant_cells)
                        print('Extended excel files')
                        
                    except Exception as e:
                        self.errors.append({'image': image_file, 'error': str(e)})
                        print(f'Error processing {image_file}: {e}')
                        #end the code here
                        json_path = f'data/refine_model_processed_thresholds/errors/{image_name}_error.json'
                        with open(json_path, 'w') as json_file:
                            json.dump(image_name, json_file)

        print(f'Starting model evaluation')
        self.model_eval(excel_files, valid_cells, predicted_cells)


# Example usage:
inform_files = '/Rebeca&Laura/inform_in_excel/'
granzyme_b_image_folder = '/UEC, CD8 and GranzymeB/'
model_path =  'data/saved_models/10e_trans_with_relax.pth'
patch_size = 256

inference = inFormModelEval(inform_files, granzyme_b_image_folder, model_path, patch_size)
inference.run_model_val()