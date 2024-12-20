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
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, LassoSelector, RectangleSelector
from matplotlib.path import Path
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_interactions import panhandler, zoom_factory

#Need to save the info and deterimine how much of the sample of the analysis is done. Being able to find the treshold


# Connect to Dropbox
ACCESS_TOKEN = 'sl.B6YpmPE00UgRMkgbZF8cLwcnB-am969CJMCV9HaNdBa7iJUuckhfqAkbXlSu_30PICr7znknvH2Mn4nXjnDsltQ3rf7W6W63NlKadKZ3zCnNSbd625T32-KZVxbiN2WKXWrW_iivychG'
dbx = dropbox.Dropbox(ACCESS_TOKEN)

class RLEnvironment:
    def __init__(self, model, device, patch_size=256):
        self.model = model
        self.device = device
        self.patch_size = patch_size

    def reset(self, image):
        self.image = image
        self.patches = self.create_patches_no_box(image, self.patch_size)
        self.current_patch_idx = 0
        return self.patches[self.current_patch_idx]

    def step(self, action, pred_boxes, pred_scores, true_boxes):
        reward = self.calculate_reward(pred_boxes, true_boxes)
        self.current_patch_idx += 1
        done = self.current_patch_idx >= len(self.patches)
        next_state = self.patches[self.current_patch_idx] if not done else None
        return next_state, reward, done

    def calculate_reward(self, pred_boxes, true_boxes):
        reward = 0
        for pred_box in pred_boxes:
            if self.is_correctly_excluded(pred_box, true_boxes):
                reward += 1
            else:
                reward -= 1
        return reward

    def is_correctly_excluded(self, pred_box, true_boxes):
        for true_box in true_boxes:
            if self.iou(pred_box, true_box) > 0.5:
                return False
        return True

    def iou(self, box1, box2):
        # Validate bounding boxes
        if not self.is_valid_box(box1) or not self.is_valid_box(box2):
            return 0

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def is_valid_box(self, box):
        # Check if the box is a valid numpy array or list/tuple
        if isinstance(box, (list, tuple, np.ndarray)):
            if len(box) != 4 or any(coord < 0 for coord in box):
                print(f"Invalid box format or negative values: {box}")
                return False
        else:
            print(f"Box is not a list, tuple, or numpy array: {box}")
            return False
        return True

    def create_patches_no_box(self, image, patch_size):
        patches = []
        img_height, img_width = image.shape[:2]
        for i in range(0, img_height, patch_size):
            for j in range(0, img_width, patch_size):
                patch = image[i:i + patch_size, j:j + patch_size]
                patches.append((patch, i, j))
        return patches


#All the relevant info is in the
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

    def label_relevant_cells(self, relevant_rows, auto_98):
        aut_nuclei = 'Nucleus Autofluorescence Mean (Normalized Counts, Total Weighting)'
        cd8_mem = 'Membrane CD8 (Opal 570) Mean (Normalized Counts, Total Weighting)'
        cd8_cyt = 'Cytoplasm CD8 (Opal 570) Mean (Normalized Counts, Total Weighting)'
        gb_mem = 'Membrane Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)'
        gb_cyt = 'Cytoplasm Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)'
        gb_ent = 'Entire Cell Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)'

        relevant_cells = relevant_rows.copy()
        relevant_cells['Label'] = 'None'

        relevant_cells[aut_nuclei] = pd.to_numeric(relevant_cells[aut_nuclei], errors='coerce')
        relevant_cells[cd8_mem] = pd.to_numeric(relevant_cells[cd8_mem], errors='coerce')
        relevant_cells[cd8_cyt] = pd.to_numeric(relevant_cells[cd8_cyt], errors='coerce')
        relevant_cells[gb_mem] = pd.to_numeric(relevant_cells[gb_mem], errors='coerce')
        relevant_cells[gb_cyt] = pd.to_numeric(relevant_cells[gb_cyt], errors='coerce')
        relevant_cells[gb_ent] = pd.to_numeric(relevant_cells[gb_ent], errors='coerce')

        # Add the labels to the relevant cells dataframe
        for index, row in relevant_cells.iterrows():
            if row[gb_ent] > 0:
                #if row[aut_nuclei] > auto_98:
                if row[cd8_mem] > row[cd8_cyt]:
                    if row[gb_cyt] > row[cd8_cyt]:
                        relevant_cells.at[index, 'Label'] = 'gb'
            else:
                relevant_cells.at[index, 'Label'] = 'None'

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

    def generate_bounding_boxes(self, patch, properties, pred_boxes, pred_scores, criteria_labels, i_offset, j_offset):
        boxes = []
        #fig, ax = plt.subplots(figsize=(10, 10))
        #ax.imshow(patch)

        margin = 5  # Large margin to accommodate the possible bounding boxes

        score_treshold = 0.5

        for pred_box, score in zip(pred_boxes, pred_scores):
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
                
                if expanded_minc <= x_position <= expanded_maxc and expanded_minr <= y_position <= expanded_maxr:
                    #print("yess")
                    #ax.plot(x_position - j_offset, y_position - i_offset, 'o', color='black', markersize=4)
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


        #plt.axis('off')
        #plt.tight_layout()
        #plt.show()
        return boxes

    def full_processing(self, image, patch_size, criteria_labels, image_name):
        patches = self.create_patches_no_box(image, patch_size)
        patches = self.run_inference_on_patches(patches)
        patches_with_boxes = []

        for patch, pred_boxes, pred_scores, i, j in patches:
            properties = self.process_image(patch)
            print('Generating bounding boxes...')
            boxes = self.generate_bounding_boxes(patch, properties, pred_boxes, pred_scores, criteria_labels, i, j)
            patches_with_boxes.append((patch, boxes, i, j))

        
        all_boxes = [box for _, boxes, _, _ in patches_with_boxes for box in boxes]
        #self.plot_image_with_boxes(new_image, all_boxes, title="Reconstructed Image with Bounding Boxes")
        print('Number of boxes:', len(all_boxes))
        print('Type of all_boxes:', type(all_boxes))

        # Do the manual change of the boxes
        # Reinforcment learning to improve the model
        # Save the information of the sample as well as like a folder of the ones that are already done, to not repeat the process
        # Particularlly because there are too many images for a single sample (ex, Opl 221_8, 61 images)
            # Do the manual change of the boxes
        if self.manual:
            valid_boxes = self.manual_elimination(image, all_boxes)
            num_epochs = 3
            self.reinforcement_learning(self.model, image, all_boxes, valid_boxes, self.device, num_epochs)
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

                # Extract only the bounding box coordinates from true_boxes
                true_boxes_coords = [box['Bounding Box'] for box in true_boxes]

                # Validate pred_boxes before proceeding
                valid_pred_boxes = [box for box in pred_boxes if self.rl_env.is_valid_box(box)]

                # Define your action based on the model prediction (this part is simplified)
                action = valid_pred_boxes
                next_state, reward, done = self.rl_env.step(action, valid_pred_boxes, pred_scores, true_boxes_coords)
                # Update your Q-values or policy network based on the reward (simplified here)
                state = next_state

    def full_processing(self, image, patch_size, criteria_labels, image_name):
        patches = self.create_patches_no_box(image, patch_size)
        patches = self.run_inference_on_patches(patches)
        patches_with_boxes = []

        for patch, pred_boxes, pred_scores, i, j in patches:
            properties = self.process_image(patch)
            print('Generating bounding boxes...')
            boxes = self.generate_bounding_boxes(patch, properties, pred_boxes, pred_scores, criteria_labels, i, j)
            patches_with_boxes.append((patch, boxes, i, j))

        all_boxes = [box for _, boxes, _, _ in patches_with_boxes for box in boxes]
        print('Number of boxes:', len(all_boxes))
        print('Type of all_boxes:', type(all_boxes))

        if self.manual:
            valid_boxes = self.manual_elimination(image, all_boxes)
            self.reinforcement_learning(image, valid_boxes, num_epochs=3)
            return valid_boxes

        new_image = self.reconstruct_image(image, patches_with_boxes, patch_size)
        return all_boxes
    '''
    Add the reinforcement learning to improve the model. Get the other basics functions first and then attempt to fix this one


    def reinforcement_learning(self, model, image, original_boxes, updated_boxes, device, num_epochs):
        pseudo_labels = []
        for box in original_boxes:
            if box not in updated_boxes:
                label = 0  # Incorrect detection
            else:
                label = 1  # Correct detection
            pseudo_labels.append((box['Bounding Box'], label))

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0

            for patch, target in self.get_patches_and_targets(image, pseudo_labels):
                optimizer.zero_grad()
                output = model([patch.to(device)])
                # Compute loss using only the classifier head's output
                loss_dict = model.roi_heads.box_predictor(output[0], target)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()
                epoch_loss += losses.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(pseudo_labels)}")

        # Save the updated model
        torch.save(model.state_dict(), 'reinforced_model.pth')

    def get_patches_and_targets(self, image, pseudo_labels):
        patches_with_labels = []
        rows, cols, _ = image.shape

        patch_size = 256

        def normalize_bbox(bbox, rows, cols):
            x_min, y_min, x_max, y_max = bbox
            return [x_min / cols, y_min / rows, x_max / cols, y_max / rows]

        for box, label in pseudo_labels:
            x_min, y_min, x_max, y_max = box
            patch = image[y_min:y_max, x_min:x_max]  # Fixed slicing order
            patch_boxes = [[0, 0, x_max - x_min, y_max - y_min]]
            patch_labels = [label]
            patch_boxes = [normalize_bbox(bbox, y_max - y_min, x_max - x_min) for bbox in patch_boxes]

            augmented = self.get_transforms()(image=patch, bboxes=patch_boxes, labels=patch_labels)
            patch = augmented['image']
            patch = patch.clone().detach().float() / 255.0  # Normalize the image to [0, 1]

            patch_boxes = torch.as_tensor(augmented['bboxes'], dtype=torch.float32)
            patch_labels = torch.as_tensor(augmented['labels'], dtype=torch.int64)

            target = {"boxes": patch_boxes, "labels": patch_labels}

            patches_with_labels.append((patch, target))

        return patches_with_labels
    '''


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
            if perimeter == 0:
                circularity = 0
            else:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            circularities.append(circularity)

        intensity_threshold = np.percentile(nuclei_intensities, 10)

        lymphocytes_with_dab = [
            prop.label for prop, nuclei_intensity, dab_intensity, circularity in zip(properties, nuclei_intensities, dab_intensities, circularities)
            if nuclei_intensity < intensity_threshold and dab_intensity > 0 and circularity > 0.55
        ]

        lymphocyte_dab_mask = np.isin(masks, lymphocytes_with_dab)
        #fig, ax = plt.subplots(figsize=(10, 10))
        #ax.imshow(input_image, cmap='gray')
        #ax.imshow(lymphocyte_dab_mask, alpha=0.5, cmap='jet')
        #ax.set_title('Identified Lymphocytes with DAB')
        #ax.axis('off')
        #plt.show()

        filtered_properties = [prop for prop in properties if prop.label in lymphocytes_with_dab]
        #filtered_properties

        return filtered_properties

    def run(self, example):
        #Dosen't use the auto_98 value
        inform_files = self.list_files_in_folder(self.inform_files)
        image_files = self.list_files_in_folder(self.granzyme_b_image_folder)

        for inform_file in inform_files:
            if inform_file.endswith('.xlsx'):
                inform_path = self.inform_files + inform_file
                inform_excel = self.read_excel_from_dropbox(inform_path)
                name_sample = inform_file.split('.xlsx')[0]
                if name_sample != example:
                    continue
                auto_98 = self.global_inform_values(inform_excel)
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
                        criteria_labels = self.label_relevant_cells(relevant_rows, auto_98)
                        gb_image = self.read_image_from_dropbox(image_path)
                        patch_size = 256
                        #use all the cells not the criteria ones
                        all_boxes = self.full_processing(gb_image, patch_size, criteria_labels, image_name)
                        valid_cells.append(all_boxes)
                        #{'image': image_file, 'boxes': all_boxes})
                    except Exception as e:
                        self.errors.append({'image': image_file, 'error': str(e)})
                        print(f'Error processing {image_file}: {e}')
                        json_path = f'new_simpie_flexible/error/{image_name}_error.json'
                        with open(json_path, 'w') as json_file:
                            json.dump(image_name, json_file)
                print('Done!')
                print(valid_cells)
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

