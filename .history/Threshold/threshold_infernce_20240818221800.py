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



# Connect to Dropbox
ACCESS_TOKEN = 'sl.B7OWaZUbsKNjc76nyrtCZnl-YDH8sDCd_lJ6BDiqQ-FMPlzlLaCmcVlZ1b2HPXbzRvOFgfbICiIlNoxTaG7dJtgPvZDRnldNGGMBd9j3l6s4mGk92NiKsiia02IEdm_R9HQH08nAAq90'
dbx = dropbox.Dropbox(ACCESS_TOKEN)


#All the relevant info is in the
class InferenceWithThreshold:
    def __init__(self, inform_files, granzyme_b_image_folder, processed_data_folder, model_path, patch_size, manual= False):
        self.inform_files = inform_files
        self.granzyme_b_image_folder = granzyme_b_image_folder
        self.processed_data_folder = processed_data_folder
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
        return positive_gb_cells

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

        for patch, pred_boxes, pred_scores, i, j in patches:
            #properties = self.process_image(patch)
            print('Generating bounding boxes...')
            boxes = self.generate_bounding_boxes(patch, pred_boxes, pred_scores, criteria_labels, i, j)
            patches_with_boxes.append((patch, boxes, i, j))

        all_boxes = [box for _, boxes, _, _ in patches_with_boxes for box in boxes]
        print('Number of boxes:', len(all_boxes))
        print('Type of all_boxes:', type(all_boxes))

        if self.manual:
            valid_boxes = self.manual_elimination(image, all_boxes)
            exlude_boxes = [box for box in all_boxes if box not in valid_boxes]
            return valid_boxes

        #new_image = self.reconstruct_image(image, patches_with_boxes, patch_size)
        #self.plot_image_with_boxes(new_image, all_boxes, title="Reconstructed Image with Bounding Boxes")
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


    def run(self, example):
        human_in_loop = True
        inform_files = self.list_files_in_folder(self.inform_files)
        image_files = self.list_files_in_folder(self.granzyme_b_image_folder)
        errors = os.listdir('/Users/rebeca/Documents/GitHub/SHY_deep_learning/data/processed_thresholds/errors')
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

                    if f'{image_name}' in errors:
                        print(f'Skipping image {image_name} due to previous error')
                        continue

                    processed_data = self.load_processed_data(name_sample, image_name)
                    # load the name of the errors folder
                    if processed_data:
                        print(f'The image {image_name} was already processed!')
                        valid_cells.extend(processed_data)
                    else:

                        try:
                            image_path = self.granzyme_b_image_folder + image_file
                            print(f'Processing image: {image_name}')
                            relevant_rows = self.relevant_rows(image_name, inform_excel)
                            criteria_labels = self.label_relevant_cells(relevant_rows)
                            gb_image = self.read_image_from_dropbox(image_path)
                            all_boxes = self.full_processing(gb_image, self.patch_size, criteria_labels, image_name)
                            valid_cells.extend(all_boxes)
                            if human_in_loop:
                                self.human_in_the_loop_update(gb_image, all_boxes, self.patch_size)
                            self.save_processed_data(name_sample, image_name, all_boxes)
                            print(f'The image {image_name} was saved')
                        except Exception as e:
                            self.errors.append({'image': image_file, 'error': str(e)})
                            print(f'Error processing {image_file}: {e}')
                            #end the code here
                            json_path = f'data/processed_thresholds/errors/{image_name}_error.json'
                            with open(json_path, 'w') as json_file:
                                json.dump(image_name, json_file)

                print(f'Done processing sample: {name_sample}')
                self.calculate_threshold(valid_cells, inform_excel)

    def get_train_transforms(self):
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

        # Create patches using torchvision.transforms.functional
    def create_patches(self, image, boxes, patch_size):
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
        print(f'Number of patches: {len(patches)}')
        return patches
    
    def log_metrics(self, loss):
        log_file = 'refine_model_copy_20e.json'
        import datetime

        log_entry = {
            'loss': loss,
            'timestamp': str(datetime.now())
            }

        # Check if the log file exists
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(log_entry)

        # Write back to the file
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=4)

        print('Loss of the model logge successfully!')
    
    def human_in_the_loop_update(self, image, valid_cells, patch_size):
        """
        This method allows manual inspection and elimination of bounding boxes before updating the model.
        """
        print('Starting human-in-the-loop update...')
        logging.basicConfig(
            filename='hitl_model.log', 
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='a'  
            )
        logging.info('Starting human-in-the-loop update...')
        # Log in the metrics, particualrlly the loss of the md

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005) #lr = 0.0005 for 10 epochs fine tuning

        #patches = self.create_patches_no_box(image, patch_size)

        # Generate bounding boxes using the provided criteria
        boxes = [valid_cell['Bounding Box'] for valid_cell in valid_cells]
        #print(boxes)
        #print(f'Number of valid cells: {len(boxes)}')

        transform = self.get_train_transforms()

        # Create patches using these bounding boxes
        image_patches, target_patches = [], []
        for patch, patch_boxes in self.create_patches(image, boxes, patch_size):
            # Create target labels, apply transformations, etc.
            if len(patch_boxes) == 0:
                patch_boxes = torch.zeros((0, 4), dtype=torch.float32)
                patch_labels = torch.zeros((0,), dtype=torch.int64)
                masks = torch.zeros((0, patch.shape[0], patch.shape[1]), dtype=torch.uint8)
            else:
                patch_boxes = torch.as_tensor(patch_boxes, dtype=torch.float32)
                patch_labels = torch.as_tensor([1 for _ in patch_boxes], dtype=torch.int64)
                masks = torch.zeros((0, patch.shape[0], patch.shape[1]), dtype=torch.uint8)

            target = {"boxes": patch_boxes, "labels": patch_labels, "masks": masks}

            augmented = transform(image=patch, bboxes=patch_boxes, labels=patch_labels)
            patch = augmented['image']
            patch_boxes = augmented['bboxes']
            patch_labels = augmented['labels']
            patch = patch.clone().detach().float() / 255.0  # Normalize the image to [0, 1]

            patch_boxes = torch.as_tensor(patch_boxes, dtype=torch.float32)
            patch_labels = torch.as_tensor(patch_labels, dtype=torch.int64)

            target = {"boxes": patch_boxes, "labels": patch_labels, "masks": masks}
            image_patches.append(patch)
            target_patches.append(target)


        # Update model using the remaining valid patches
        #epochs = 3
    
        #for epoch in range(epochs):
        loss = self.train_single_epoch(image_patches, target_patches, optimizer)
        print(f'Loss  after human-in-the-loop update for image: {loss}')
        logging.info(f'Loss after human-in-the-loop update for image: {loss}')
        
        return loss

    def train_single_epoch(self, image_patches, target_patches, optimizer):
        print('Training started')
        self.model.train()
        epoch_loss = 0
        for image, target in zip(image_patches, target_patches):
            if target["boxes"].numel() == 0:
                continue
            optimizer.zero_grad()
            loss_dict = self.model([image.to(self.device)], [target])
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()


        return epoch_loss

    def calculate_threshold(self, valid_cells, inform_data):
        valid_cell_ids = [cell['Cell ID'] for cell in valid_cells]
        inform_data['True Label'] = inform_data['Cell ID'].apply(lambda x: 1 if x in valid_cell_ids else 0)

        true_labels = inform_data['True Label'].tolist()
        granzyme_b_values = inform_data['Entire Cell Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)'].tolist()

        # Use ROC curve to determine the optimal threshold
        fpr, tpr, thresholds = roc_curve(true_labels, granzyme_b_values)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='purple',  label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
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
inform_files = '/Rebeca&Laura/inform_in_excel/'
granzyme_b_image_folder = '/UEC, CD8 and GranzymeB/'
processed_data_folder = 'data/processed_thresholds'
model_path =  'data/saved_models/copy_20e_trans_with_relax.pth'
patch_size = 256
example_sample = 'Opal 221_8'  # Example

inference = InferenceWithThreshold(inform_files, granzyme_b_image_folder, processed_data_folder, model_path, patch_size)
inference.run(example_sample)