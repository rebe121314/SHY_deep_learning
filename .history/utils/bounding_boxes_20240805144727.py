# bounding_boxes.py

import matplotlib.patches as mpatches

def generate_bounding_boxes(patch, properties, pred_boxes, pred_scores, criteria_labels, i_offset, j_offset, score_threshold=0.75):
    boxes = []
    margin = 5  # Large margin to accommodate the possible bounding boxes

    for pred_box, score in zip(pred_boxes, pred_scores):
        if score < score_threshold:
            continue
        x_min, y_min, x_max, y_max = pred_box
        rect = mpatches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='blue', facecolor='none', linestyle='dashed'
        )
        expanded_minc = max(0, x_min - margin) + j_offset
        expanded_maxc = x_max + margin + j_offset
        expanded_minr = max(0, y_min - margin) + i_offset
        expanded_maxr = y_max + margin + i_offset

        for _, row in criteria_labels.iterrows():
            x_position = row['Cell X Position']
            y_position = row['Cell Y Position']
            
            if expanded_minc <= x_position <= expanded_maxc and expanded_minr <= y_position <= expanded_maxr:
                boxes.append({
                    'Cell ID': row['Cell ID'],
                    'X Position': int(x_position),
                    'Y Position': int(y_position),
                    'Bounding Box': [int(x_min + j_offset), int(y_min + i_offset), int(x_max + j_offset), int(y_max + i_offset)],
                    'Granzyme B': row['Entire Cell Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)'],
                    'Score': score
                })
    return boxes
