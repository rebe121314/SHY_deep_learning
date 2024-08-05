# reinforcement_learning.py

import torch.optim as optim

def reinforcement_learning(model, image, original_boxes, updated_boxes, device, num_epochs=3):
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

        for patch, target in get_patches_and_targets(image, pseudo_labels):
            optimizer.zero_grad()
            output = model([patch.to(device)])
            loss_dict = model.roi_heads.box_predictor(output[0], target)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(pseudo_labels)}")

    torch.save(model.state_dict(), 'reinforced_model.pth')

def get_patches_and_targets(image, pseudo_labels):
    patches_with_labels = []
    rows, cols, _ = image.shape

    patch_size = 256

    def normalize_bbox(bbox, rows, cols):
        x_min, y_min, x_max, y_max = bbox
        return [x_min / cols, y_min / rows, x_max / cols, y_max / rows]

    for box, label in pseudo_labels:
        x_min, y_min, x_max, y_max = box
        patch = image[y_min:y_max, x_min:x_max]
        patch_boxes = [[0, 0, x_max - x_min, y_max - y_min]]
        patch_labels = [label]
        patch_boxes = [normalize_bbox(bbox, y_max - y_min, x_max - x_min) for bbox in patch_boxes]

        augmented = A.Compose([ToTensorV2()])(image=patch, bboxes=patch_boxes, labels=patch_labels)
        patch = augmented['image']

        patch_boxes = torch.as_tensor(augmented['bboxes'], dtype=torch.float32)
        patch_labels = torch.as_tensor(augmented['labels'], dtype=torch.int64)

        target = {"boxes": patch_boxes, "labels": patch_labels}

        patches_with_labels.append((patch, target))

    return patches_with_labels
