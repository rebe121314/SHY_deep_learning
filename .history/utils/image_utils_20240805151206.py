import numpy as np
from skimage import img_as_ubyte, measure
from skimage.color import rgb2hed, hed2rgb
from skimage.exposure import rescale_intensity, equalize_adapthist
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from cellpose import models

class ImageProcessor:
    @staticmethod
    def create_patches_no_box(image, patch_size):
        patches = []
        img_height, img_width = image.shape[:2]
        for i in range(0, img_height, patch_size):
            for j in range(0, img_width, patch_size):
                patch = image[i:i + patch_size, j:j + patch_size]
                patches.append((patch, i, j))
        return patches

    @staticmethod
    def color_separate(ihc_rgb):
        ihc_hed = rgb2hed(ihc_rgb)
        null = np.zeros_like(ihc_hed[:, :, 0])
        ihc_h = img_as_ubyte(hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1)))
        ihc_e = img_as_ubyte(hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1)))
        ihc_d = img_as_ubyte(hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1)))
        h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1), in_range=(0, np.percentile(ihc_hed[:, :, 0], 99)))
        d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1), in_range=(0, np.percentile(ihc_hed[:, :, 2], 99)))
        zdh = img_as_ubyte(np.dstack((null, d, h)))
        return ihc_h, ihc_e, ihc_d, zdh

    @staticmethod
    def reconstruct_image(original_image, patches, patch_size):
        fig, ax = plt.subplots(1, figsize=(12, 12))
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
        return original_image

    @staticmethod
    def plot_image_with_boxes(image, boxes, pred_boxes=None, pred_scores=None, title="Image with Bounding Boxes"):
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
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(input_image, cmap='gray')
        ax.imshow(lymphocyte_dab_mask, alpha=0.5, cmap='jet')
        ax.set_title('Identified Lymphocytes with DAB')
        ax.axis('off')
        plt.show()

        filtered_properties = [prop for prop in properties if prop.label in lymphocytes_with_dab]

        return filtered_properties
