# modified from https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/utils/box_utils.py

import collections
import torch
from typing import List
import math


Box = collections.namedtuple('Box', ['height', 'width'])
ImageDimensions = collections.namedtuple('ImageDimensions', ['height', 'width'])
BoxSizes = collections.namedtuple('BoxSizes', ['min', 'max'])

PriorSpec = collections.namedtuple('PriorSpec', ['feature_map_size_height', 'feature_map_size_width', 'shrinkage_height',
                                            'shrinkage_width',
                                            'box_sizes', 'aspect_ratios'])


def generate_priors(specs: List[PriorSpec], image_size: ImageDimensions, clamp=True) -> torch.Tensor:
    """Generate Prior Boxes.

    It returns the center, height and width of the priors. The values are relative to the image size
    Args:
        specs: PriorSpecs about the shapes of sizes of prior boxes. i.e.
        image_size: image size.
        clamp: if true, clamp the values to make fall between [0.0, 1.0]
    Returns:
        priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
            are relative to the image size.
    """
    priors = []
    for spec in specs:
        scale_height = image_size.height / spec.shrinkage_height
        scale_width = image_size.width / spec.shrinkage_width
        for j in range(spec.feature_map_size_height):
            for i in range(spec.feature_map_size_width):
                x_center = (i + 0.5) / scale_width
                y_center = (j + 0.5) / scale_height

                # small sized square box
                size = spec.box_sizes.min
                h = size / image_size.height
                w = size / image_size.width
                priors.append([
                    x_center,
                    y_center,
                    w,
                    h
                ])

                # big sized square box
                size = math.sqrt(spec.box_sizes.max * spec.box_sizes.min)
                h = size / image_size.height
                w = size / image_size.width

                for ratio in spec.aspect_ratios:
                    ratio = math.sqrt(ratio)
                    priors.append([
                        x_center,
                        y_center,
                        w,         # w / ratio
                        h * ratio
                    ])

                
                # change h/w ratio of the small sized box
                size = spec.box_sizes.min
                h = size / image_size.height
                w = size / image_size.width
                
                for ratio in spec.aspect_ratios:
                    ratio = math.sqrt(ratio)
                    priors.append([
                        x_center,
                        y_center,
                        w,         # w / ratio
                        h * ratio
                    ])

    priors = torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors


def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).
    
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, h, w]]. All the values
            are relative to the image size.
    """
    locations = locations.to("cpu")
    priors = priors.to("cpu")
    # priors can have one dimension less.
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], dim=locations.dim() - 1)

def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def center_form_to_corner_form(locations):
    return torch.cat([locations[..., :2] - locations[..., 2:]/2,
                     locations[..., :2] + locations[..., 2:]/2], locations.dim() - 1) 


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            current_box.unsqueeze(0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def nms(box_scores, iou_threshold=None, top_k=-1, candidate_size=200):
    return hard_nms(box_scores, iou_threshold, top_k, candidate_size=candidate_size)
