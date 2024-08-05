# inference.py

from utils.dropbox_utils import list_files_in_folder, read_image_from_dropbox, read_excel_from_dropbox
from utils.image_utils import create_patches, color_separate
from utils.plot_utils import plot_image_with_boxes
from utils.data_processing import save_processed_data, load_processed_data
from utils.bounding_boxes import generate_bounding_boxes
from utils.reinforcement_learning import reinforcement_learning

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

        for index, row in relevant_cells.iterrows():
            if row[gb_ent] > 0 and row['Phenotype'] != 'Tumor cell':
                if row[cd8_mem] > row[cd8_cyt]:
                    if row[gb_cyt] > row[cd8_cyt]:
                        relevant_cells.at[index, 'Label'] = 'gb'
            else:
                relevant_cells.at[index, 'Label'] = 'None'

        positive_gb_cells = relevant_cells[relevant_cells['Label'] == 'gb']
        return positive_gb_cells

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

    def full_processing(self, image, patch_size, criteria_labels, image_name):
        patches = create_patches(image, patch_size)
        patches = self.run_inference_on_patches(patches)
        patches_with_boxes = []

        for patch, pred_boxes, pred_scores, i, j in patches:
            properties = self.process_image(patch)
            boxes = generate_bounding_boxes(patch, properties, pred_boxes, pred_scores, criteria_labels, i, j)
            patches_with_boxes.append((patch, boxes, i, j))

        all_boxes = [box for _, boxes, _, _ in patches_with_boxes for box in boxes]

        if self.manual:
            valid_boxes = self.manual_elimination(image, all_boxes)
            reinforcement_learning(self.model, image, all_boxes, valid_boxes, self.device, num_epochs=3)
            return valid_boxes

        return all_boxes

    def process_image(self, gb):
        H, _, D, _ = color_separate(gb)
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

        filtered_properties = [prop for prop in properties if prop.label in lymphocytes_with_dab]
        return filtered_properties

    def run(self, example):
        inform_files = list_files_in_folder(self.inform_files)
        image_files = list_files_in_folder(self.granzyme_b_image_folder)
        errors = os.listdir(f'{self.processed_data_folder}/errors')
        errors = [f.split('_error.json')[0] for f in errors]

        for inform_file in inform_files:
            if inform_file.endswith('.xlsx'):
                inform_path = self.inform_files + inform_file
                inform_excel = read_excel_from_dropbox(inform_path)
                name_sample = inform_file.split('.xlsx')[0]
                if name_sample != example:
                    continue

                relevant_images = [f for f in image_files if name_sample in f and 'Granzyme' in f]

                valid_cells = []
                processed_sample_folder = os.path.join(self.processed_data_folder, name_sample)
                if not os.path.exists(processed_sample_folder):
                    os.makedirs(processed_sample_folder)

                for image_file in relevant_images:
                    image_name = image_file.split('_Granzyme')[0]

                    if f'{image_name}' in errors:
                        continue

                    processed_data = load_processed_data(name_sample, image_name, self.processed_data_folder)
                    if processed_data:
                        valid_cells.extend(processed_data)
                    else:
                        try:
                            image_path = self.granzyme_b_image_folder + image_file
                            relevant_rows = self.relevant_rows(image_name, inform_excel)
                            criteria_labels = self.label_relevant_cells(relevant_rows)
                            gb_image = read_image_from_dropbox(image_path)
                            all_boxes = self.full_processing(gb_image, self.patch_size, criteria_labels, image_name)
                            valid_cells.extend(all_boxes)
                            save_processed_data(name_sample, image_name, all_boxes, self.processed_data_folder)
                        except Exception as e:
                            self.errors.append({'image': image_file, 'error': str(e)})
                            json_path = f'{self.processed_data_folder}/errors/{image_name}_error.json'
                            with open(json_path, 'w') as json_file:
                                json.dump(image_name, json_file)

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
if __name__ == '__main__':
    from config import INFORM_FILES, GRANZYME_B_IMAGE_FOLDER, PROCESSED_DATA_FOLDER, MODEL_PATH, PATCH_SIZE, EXAMPLE_SAMPLE
    inference = InferenceWithThreshold(INFORM_FILES, GRANZYME_B_IMAGE_FOLDER, PROCESSED_DATA_FOLDER, MODEL_PATH, PATCH_SIZE)
    optimal_threshold, inform_data = inference.run(EXAMPLE_SAMPLE)
