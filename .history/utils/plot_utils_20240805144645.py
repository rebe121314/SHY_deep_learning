# plot_utils.py

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_image_with_boxes(image, boxes, title="Image with Bounding Boxes"):
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
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
