# utils/model_utils.py
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes, model_path=None, device=None):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    model.roi_heads.mask_predictor = None
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    if device:
        model.to(device)
    return model
