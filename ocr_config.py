# modified from https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/ssd/config/mobilenetv1_ssd_config.py

import numpy as np

from utils import ImageDimensions, PriorSpec, BoxSizes, generate_priors

image_size = ImageDimensions(600, 375)
image_size_priors = ImageDimensions(375, 600)
image_mean = np.array([127.5, 127.5, 127.5])
image_std = 128.5
iou_threshold = 0.50
center_variance = 0.1
size_variance = 0.2

specs = [
    PriorSpec(24, 38, 16, 16, BoxSizes(14, 30), [3]),
    PriorSpec(12, 19, 31, 31, BoxSizes(30, 45), [3]),
]

priors = generate_priors(specs, image_size_priors)
scale_vector = np.array([image_size.width, image_size.height, image_size.width, image_size.height])
