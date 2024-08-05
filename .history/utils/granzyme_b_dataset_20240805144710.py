# granzyme_b_dataset.py

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from albumentations.pytorch import ToTensorV2
import albumentations as A

class GranzymeBDataset(Dataset):
    def __init__(self, image_dir, labels_dir, transform, patch_size=256, val=False):
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.patch_size = patch_size
        self.val = val
        self.images, self.labels = self._load_images_and_labels()

    def _list_files_in_folder(self, folder_path):
        # Implement list_files_in_folder function here
        pass

    def _load_images_and_labels(self):
        label_files = self._list_files_in_folder(self.labels_dir)
        if self.val:
            label_files = label_files[:5]
        else:
            label_files = label_files[6:]
        image_files = set(self._list_files_in_folder(self.image_dir))

        images, labels = [], []
        for label_file in label_files:
            sample_name = label_file.replace('_labels.json', '')
            img_name = sample_name + '_Granzyme B_path_view.tif'
            if img_name in image_files:
                images.append(img_name)
                labels.append(label_file)
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        label_name = self.labels[idx]
        img_path = f"{self.image_dir}/{img_name}"
        image = read_image_from_dropbox(img_path)  # Implement read_image_from_dropbox function here
        label_path = f"{self.labels_dir}/{label_name}"
        label_data = load_annotation(label_path)  # Implement load_annotation function here

        boxes = [box["Bounding Box"] for box in label_data]
        patches = create_patches(image, self.patch_size)

        image_patches, target_patches = [], []
        for patch, patch_boxes in patches:
            patch_labels = np.array([1 for _ in patch_boxes])
            patch_boxes = torch.as_tensor(patch_boxes, dtype=torch.float32)
            patch_labels = torch.as_tensor(patch_labels, dtype=torch.int64)
            masks = torch.zeros((len(patch_boxes), patch.shape[0], patch.shape[1]), dtype=torch.uint8)

            target = {"boxes": patch_boxes, "labels": patch_labels, "masks": masks}

            if self.transform:
                augmented = self.transform(image=patch, bboxes=patch_boxes, labels=patch_labels)
                patch = augmented['image']
                patch_boxes = torch.as_tensor(augmented['bboxes'], dtype=torch.float32)
                patch_labels = torch.as_tensor(augmented['labels'], dtype=torch.int64)

            target = {"boxes": patch_boxes, "labels": patch_labels}

            image_patches.append(patch)
            target_patches.append(target)

        return image_patches, target_patches
