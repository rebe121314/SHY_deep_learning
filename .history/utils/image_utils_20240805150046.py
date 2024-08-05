# utils/image_utils.py
import numpy as np
from skimage import img_as_ubyte, measure
from skimage.color import rgb2hed, hed2rgb
from skimage.exposure import rescale_intensity, equalize_adapthist
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

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

def plot_image_with_boxes(image, boxes, pred_boxes=None, pred_scores=None, title="Image with Bounding Boxes"):
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

