import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
import torch
import torch.optim as optim
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def load_model(model_path):
    return torch.load(model_path)


def get_model_instance_segmentation(num_classes, model_name='resnet_50_fpn'):
    if model_name == 'resnet_50_fpn':
        # load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)
        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)

    elif model_name == 'resnet_101_fpn':
        model = get_mask_rcnn_resnet_101_fpn(num_classes)

    return model


def get_mask_rcnn_resnet_101_fpn(num_classes):
    backbone = resnet_fpn_backbone(
        'resnet101', pretrained=True, trainable_layers=3)
    model = MaskRCNN(backbone, num_classes,
                     box_detections_per_img=200)
    return model


def get_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-4)
    # optimizer = torch.optim.Adam(params, lr=1e-5)
    return optimizer


def get_scheduler(optimizer):
    def lambda_lr(epoch):
        # if epoch < 60:
        if epoch < 50:
            return 1.0
        else:
            return 0.1
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)
    return scheduler
