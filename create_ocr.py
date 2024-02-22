import torch
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d
from torch import nn
from mobilenet_v2 import MobileNetV2

from ocr import OCR, OCRPredictor
import ocr_config as config


def reg_conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    ReLU = nn.ReLU
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        BatchNorm2d(in_channels),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )
def class_conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    ReLU = nn.ReLU 
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        BatchNorm2d(in_channels),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        BatchNorm2d(in_channels),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        BatchNorm2d(in_channels),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )



def create_mobilenetv2_ocr(num_classes, width_mult=1.0, use_batch_norm=True, is_test=False):
    base_net = MobileNetV2(width_mult=width_mult, use_batch_norm=use_batch_norm).features
    source_layer_indexes = [

        (14, 'conv', 3),
        19,
    ]
    regression_headers = ModuleList([
        reg_conv2d(in_channels=round(576 * width_mult), out_channels=3 * 4,
                        kernel_size=3, padding=1),
        reg_conv2d(in_channels=1280, out_channels=3 * 4, kernel_size=3, padding=1),
    ])

    classification_headers = ModuleList([
        class_conv2d(in_channels=round(576 * width_mult), out_channels=3 * num_classes, kernel_size=3, padding=1),
        class_conv2d(in_channels=1280, out_channels=3 * num_classes, kernel_size=3, padding=1),
    ])

    return OCR(num_classes, base_net, source_layer_indexes,
               classification_headers, regression_headers, is_test=is_test, config=config)


def create_mobilenetv2_ocr_predictor(net, candidate_size=200, device=torch.device('cpu')):
    predictor = OCRPredictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          device=device)
    return predictor
