import dropbox
import io
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio, img_as_ubyte, measure
from skimage.color import rgb2hed, hed2rgb
from skimage.exposure import rescale_intensity, equalize_adapthist
import json
from tqdm import tqdm
import matplotlib.patches as mpatches
from cellpose import models
import dropbox
import io
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio, img_as_ubyte, measure
from skimage.color import rgb2hed, hed2rgb
from skimage.exposure import rescale_intensity, equalize_adapthist
import pyclesperanto_prototype as cle
from skimage.measure import regionprops
import json
import python_calamine
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
dbx = dropbox.Dropbox(ACCESS_TOKEN)

class LabelGeneration:
    def __init__(self, inform_files, granzyme_b_image_folder, labels_folder, simpie_files):
        self.inform_files = inform_files
        self.granzyme_b_image_folder = granzyme_b_image_folder
        self.labels_folder = labels_folder
        self.simpie_files = simpie_files
        self.errors = []

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
                    if row[gb_cyt] >= row[cd8_cyt]:
                        relevant_cells.at[index, 'Label'] = 'gb'
                    else:
                        #gbct_cd8 += 1
                        relevant_cells.at[index, 'Label'] = 'gbct_cd8'
                else:
                    #not_gb += 1
                    relevant_cells.at[index, 'Label'] = 'not_gb'
            else:
                #not_in_phen += 1
                relevant_cells.at[index, 'Label'] = 'not_phen'

        positive_gb_cells = relevant_cells[relevant_cells['Label'] == 'gb']
        return positive_gb_cells
    
    def simpie_cells(self, simpie_file, image_name):
        simpie_cells = pd.DataFrame()
        for index, row in simpie_file.iterrows():
            if image_name in row['Sample Name']:
                simpie_cells = simpie_cells._append(row)
        return simpie_cells

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
    
    def create_patches_no_box(self, image, patch_size):
        patches = []
        img_height, img_width = image.shape[:2]
        for i in range(0, img_height, patch_size):
            for j in range(0, img_width, patch_size):
                patch = image[i:i + patch_size, j:j + patch_size]
                patches.append((patch, i, j))
        return patches

    def reconstruct_image(self, original_image, patches, patch_size):
        fig, ax = plt.subplots(1, figsize=(12, 12))
        print('Starting reconstruction')
        ax.imshow(original_image)
        for patch, patch_boxes, i, j in patches:
            for patch_box in patch_boxes:
                bb = patch_box["Bounding Box"]
                rect = mpatches.Rectangle(
                    (bb[1], bb[0]), bb[3] - bb[1], bb[2] - bb[0],
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
            original_image[i:i + patch_size, j:j + patch_size] = patch

        print('Image after reconstruction')
        plt.imshow(original_image)

        return original_image

    def plot_image_with_boxes(self, image, boxes, pred_boxes=None, pred_scores=None, title="Image with Bounding Boxes"):
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(image)
        for box in boxes:
            bb = box["Bounding Box"]
            rect = mpatches.Rectangle(
                (bb[1], bb[0]), bb[3] - bb[1], bb[2] - bb[0],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
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

    def process_image(self, gb):
        H, _, D, _ = self.color_separate(gb)
        hematoxylin_eq = equalize_adapthist(H[:, :, 0])
        input_image = 1 - hematoxylin_eq

        model = models.Cellpose(gpu=True, model_type='nuclei')
        masks, flows, styles, diams = model.eval(input_image, diameter=None, channels=[0, 0])

        segmented_np = masks
        plt.imshow(segmented_np)

        properties = measure.regionprops(segmented_np, intensity_image=H[:, :, 0])

        return properties

    def full_processing(self, image, patch_size, criteria_labels, image_name):
        patches = self.create_patches_no_box(image, patch_size)
        patches_with_boxes = []

        for patch, i, j in patches:
            properties = self.process_image(patch)
            boxes = self.generate_bounding_boxes(patch, properties, criteria_labels, i, j)
            patches_with_boxes.append((patch, boxes, i, j))

        new_image = self.reconstruct_image(image, patches_with_boxes, patch_size)
        all_boxes = [box for _, boxes, _, _ in patches_with_boxes for box in boxes]
        self.plot_image_with_boxes(new_image, all_boxes, title="Reconstructed Image with Bounding Boxes")
        
        json_path = f'new_simpie_flexible/{image_name}_labels.json'
        with open(json_path, 'w') as json_file:
            json.dump(all_boxes, json_file)
        
        print(f'Labels for {image_name} saved to {json_path}')

    def generate_bounding_boxes(self, patch, properties, criteria_labels, i_offset, j_offset):
        boxes = []
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(patch)

        # Large margin, to accomodate the possible bounding boxes
        margin = 0

        done_points = []

        for prop in properties:
            y0, x0 = prop.centroid
            minr, minc, maxr, maxc = prop.bbox
            ax.plot(x0, y0, '.r', color='green', markersize=2)
            
            # Expand the bounding box by the margin
            expanded_minc = minc - margin
            expanded_minc = max(0, expanded_minc)  # Ensure it doesn't go below 0
            expanded_minc += j_offset
            
            expanded_maxc = maxc + margin
            expanded_maxc += j_offset
            
            expanded_minr = minr - margin
            expanded_minr = max(0, expanded_minr)  # Ensure it doesn't go below 0
            expanded_minr += i_offset
            
            expanded_maxr = maxr + margin
            expanded_maxr += i_offset

            for _, row in criteria_labels.iterrows():
                x_position = row['Cell X Position']
                y_position = row['Cell Y Position']

                if (x_position, y_position) in done_points:
                    continue
                if expanded_minc <= x_position <= expanded_maxc and expanded_minr <= y_position <= expanded_maxr:
                    ax.plot([minc, maxc, maxc, minc, minc], [minr, minr, maxr, maxr, minr], '-g', color='purple', linewidth=2)
                    ax.plot(x_position - j_offset, y_position - i_offset, 'o', color='purple', markersize=2)
                    done_points.append((x_position, y_position))
                    boxes.append({
                        'Cell ID': row['Cell ID'],
                        'X Position': int(x_position),
                        'Y Position': int(y_position),
                        'Bounding Box': [int(minc + j_offset), int(minr + i_offset), int(maxc + j_offset), int(maxr + i_offset)],
                        'Granzyme B': row['Entire Cell Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)']
                    })
                    break

        plt.axis('off')
        plt.tight_layout()
        plt.show()
        return boxes

    def fused_simpie(self):
        inform_files = self.list_files_in_folder(self.inform_files)
        image_files = self.list_files_in_folder(self.granzyme_b_image_folder)
        simpie_files = self.list_files_in_folder(self.simpie_files)

        all_files_simpie = []

        for simp in simpie_files:
            inform_path = self.simpie_files + simp
            inform_excel = self.read_excel_from_dropbox(inform_path)
            name_sample = simp.split('_all immune cell subsets.xlsx')[0]
            all_files_simpie.append(name_sample)

        for inform_file in inform_files:
            print('Start')
            if inform_file.endswith('.xlsx'):
                inform_path = self.inform_files + inform_file
                inform_excel = self.read_excel_from_dropbox(inform_path)
                name_sample = inform_file.split('.xlsx')[0]
                if name_sample in all_files_simpie:
                    auto_98 = self.global_inform_values(inform_excel)
                    print('Working on sample:', name_sample)

                    relevant_images = [f for f in image_files if name_sample in f and 'Granzyme' in f]
                    done_images = os.listdir('/Users/rebeca/Documents/GitHub/SHY_deep_learning/data/aut_label_gen')
                    done_images = [f.split('_labels.json')[0] for f in done_images]
                    errors = os.listdir('/Users/rebeca/Documents/GitHub/SHY_deep_learning/data/aut_label_gen/errors')
                    errors = [f.split('_error.json')[0] for f in errors]

                    for image_file in relevant_images:
                        try:
                            image_path = self.granzyme_b_image_folder + image_file
                            image_name = image_file.split('_Granzyme')[0]
                            if image_name in done_images or image_name in errors:
                                print(f'{image_name} already processed. Skipping...')
                                continue
                            print('Working on image:', image_name)
                            relevant_rows = self.relevant_rows(image_name, inform_excel)
                            criteria_labels = self.label_relevant_cells(relevant_rows, auto_98)
                            simpie_cells = self.simpie_cells(inform_excel, image_name)
                            simpie_cells = simpie_cells.rename(columns={'': 'Cell ID'})
                            gb_image = self.read_image_from_dropbox(image_path)
                            patch_size = 256
                            self.full_processing(gb_image, patch_size, criteria_labels, simpie_cells, image_name)
                        except Exception as e:
                            self.errors.append({'image': image_file, 'error': str(e)})
                            print(f'Error processing {image_file}: {e}')
                            json_path = f'/Users/rebeca/Documents/GitHub/SHY_deep_learning/data/aut_label_gen/errors/{image_name}_error.json'
                            with open(json_path, 'w') as json_file:
                                json.dump(image_name, json_file)
                    print('Done!')
                    print()
                else:
                    print(f'{name_sample} not in simpie files')
                    continue

    #automatic label every cell that has the basic criteria, menaing ther's going to be false positives
    def simple_criteria(self):
        inform_files = self.list_files_in_folder(self.inform_files)
        image_files = self.list_files_in_folder(self.granzyme_b_image_folder)

        for inform_file in inform_files:
            print('Start')
            if inform_file.endswith('.xlsx'):
                inform_path = self.inform_files + inform_file
                inform_excel = self.read_excel_from_dropbox(inform_path)
                name_sample = inform_file.split('.xlsx')[0]
                print('Working on sample:', name_sample)

                relevant_images = [f for f in image_files if name_sample in f and 'Granzyme' in f]
                done_images = os.listdir('/Users/rebeca/Documents/GitHub/SHY_deep_learning/data/aut_label_gen')
                done_images = [f.split('_labels.json')[0] for f in done_images]
                errors = os.listdir('/Users/rebeca/Documents/GitHub/SHY_deep_learning/data/aut_label_gen/errors')
                errors = [f.split('_error.json')[0] for f in errors]

                for image_file in relevant_images:
                    try:
                        image_path = self.granzyme_b_image_folder + image_file
                        image_name = image_file.split('_Granzyme')[0]
                        if image_name in done_images or image_name in errors:
                            print(f'{image_name} already processed. Skipping...')
                            continue
                        print('Working on image:', image_name)
                        relevant_rows = self.relevant_rows(image_name, inform_excel)
                        criteria_labels = self.label_relevant_cells(relevant_rows)
                        #simpie_cells = self.simpie_cells(inform_excel, image_name)
                        #simpie_cells = simpie_cells.rename(columns={'': 'Cell ID'})
                        gb_image = self.read_image_from_dropbox(image_path)
                        patch_size = 256
                        self.full_processing(gb_image, patch_size, criteria_labels, image_name)
                    except Exception as e:
                        self.errors.append({'image': image_file, 'error': str(e)})
                        print(f'Error processing {image_file}: {e}')
                        json_path = f'/Users/rebeca/Documents/GitHub/SHY_deep_learning/data/aut_label_gen/errors/{image_name}_error.json'
                        with open(json_path, 'w') as json_file:
                            json.dump(image_name, json_file)
                print('Done!')
                print()
            else:
                print(f'{name_sample} not in simpie files')
                continue

    def simpie_labels(self):
        simpie_files = self.list_files_in_folder(self.simpie_files)
        image_files = self.list_files_in_folder(self.granzyme_b_image_folder)

        for simp in simpie_files:
            if simp.endswith('.xlsx'):
                inform_path = self.simpie_files + simp
                inform_excel = self.read_excel_from_dropbox(inform_path)
                name_sample = simp.split('_all immune cell subsets.xlsx')[0]
                print('Working on sample:', name_sample)
                relevant_images = [f for f in image_files if name_sample in f and 'Granzyme' in f]

                for image_file in relevant_images:
                    try:
                        image_path = self.granzyme_b_image_folder + image_file
                        image_name = image_file.split('_Granzyme')[0]
                        print('Working on image:', image_name)
                        relevant_rows = self.simpie_cells(inform_excel, image_name)
                        relevant_rows = relevant_rows.rename(columns={'': 'Cell ID'})
                        gb_image = self.read_image_from_dropbox(image_path)
                        patch_size = 256
                        self.full_processing(gb_image, patch_size, relevant_rows, image_name)
                    except Exception as e:
                        self.errors.append({'image': image_file, 'error': str(e)})
                        print(f'Error processing {image_file}: {e}')
                        json_path = f'simpie_labels/Error/{image_name}_error.json'
                        with open(json_path, 'w') as json_file:
                            json.dump(image_name, json_file)


# Folder paths
inform_files = '/Rebeca&Laura/inform_in_excel/'
granzyme_b_image_folder = '/UEC, CD8 and GranzymeB/'
labels_folder = '/Lables/manual_box_label'
simpie_files = '/Rebeca&Laura/simpie/'

# Example usage:
label_gen = LabelGeneration(inform_files, granzyme_b_image_folder, labels_folder, simpie_files)
label_gen.simple_criteria()
# label_gen.test('Opal 221_1.xlsx', 'Opal 221_8_[10138,33392]_Granzyme B_path_view.tif')

#need to fix the file paths for this github folder