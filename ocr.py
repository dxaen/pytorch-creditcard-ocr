import numpy as np
from luhn import verify
import logging
import torch.nn as nn
import torch
from typing import List, Tuple
import torch.nn.functional as F

import utils
from transforms import PredictionTransform

QUICK_READ_OFFSET = 2.0
FALSE_POSITVE_TOLERANCE = 1.2
NUM_QR_DIGS = 16
NUM_QR_GROUP = 4
MIN_NUM_DIGITS = 15

def remove_false_positives(
        boxes_np: np.ndarray, labels_np: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes false positives from bounding boxes based on a standard deviation criterion.
    Args:
        boxes_np: A NumPy array of shape (N, 4) where each row represents a bounding box
                in the format [ymin, xmin, ymax, xmax].
        labels_np: A NumPy array of shape (N,) containing the corresponding labels for
                each bounding box.
    Returns:
        tuple: A tuple containing two NumPy arrays:
        - boxes_np: The filtered bounding box numpy array after removing false positives.
        - labels_np: The filtered label numpy array after removing false positives.
    """

    # Calculate median and standard deviation of ymin and ymax coordinates
    ymin_median = np.median(boxes_np[:, 1])
    ymax_median = np.median(boxes_np[:, 3])
    std_value = abs(ymax_median - ymin_median)

    outside_bounds = np.logical_or(
        boxes_np[:, 1] + std_value < ymin_median, 
        boxes_np[:, 1] - std_value > ymin_median
    )
    
    # Filter boxes and labels based on identified indices
    boxes_np = np.delete(boxes_np, np.where(outside_bounds)[0], axis=0)
    labels_np = np.delete(labels_np, np.where(outside_bounds)[0])
    return boxes_np, labels_np

def card_has_number(
    boxes_np: np.ndarray, labels_np: np.ndarray, probs_np: np.ndarray
) -> Tuple[bool, str]:
    """
    Checks if the given bounding boxes contain a number that passes the Luhn check.
    Args:
        boxes_np: A NumPy array of shape (N, 4) representing detected bounding boxes.
        labels_np: A NumPy array of shape (N,) containing corresponding labels for the boxes.
        probs_np: A NumPy array of shape (N,) containing confidence probabilities for the labels.
    Returns:
        A tuple containing a boolean indicating whether a valid number was found and 
        the string representation of the number.
    """
    string_labels: str = ""
    if not boxes_np.size:
        return False, string_labels
    # Replace any label above 9 with 0
    np.place(labels_np, labels_np > 9, 0)
    if is_quick_read(boxes_np):
        string_labels = process_quick_read(boxes_np, labels_np)
    else:
        boxes_np, labels_np = remove_false_positives(boxes_np, labels_np)
        indices_np_first_cord = np.argsort(boxes_np[:, 0])
        labels_np = np.take(labels_np, indices_np_first_cord)
        probs_np = np.take(probs_np, indices_np_first_cord)
        string_labels = ''.join(map(str, labels_np.tolist()))
    if verify(string_labels) and len(string_labels) >= MIN_NUM_DIGITS: 
        logging.debug(f"Verified {string_labels}")
        return True, string_labels
    else:
        return False, string_labels

def is_quick_read(boxes_np: np.ndarray) -> bool:
    """
    Checks if a NumPy array of bounding boxes qualifies as a "quick read" card.
    Args:
        boxes_np: A NumPy array of shape (N, 4) where each row represents a bounding box
                  in the format [ymin, xmin, ymax, xmax].
    Returns:
        bool: True if the boxes are considered a quick read, False otherwise.
    """

    num_boxes: int = boxes_np.size
    if num_boxes != NUM_QR_GROUP * NUM_QR_DIGS:
        return False

    box_centers: np.ndarray = (boxes_np[:, 1] + boxes_np[:, 3]) / 2
    box_heights: np.ndarray = np.abs(boxes_np[:, 1] - boxes_np[:, 3])

    median_center: float = np.median(box_centers)
    median_height: float = np.median(box_heights) 

    aggregate_deviation: float = np.sum(np.abs(box_centers - median_center))

    if aggregate_deviation > QUICK_READ_OFFSET * median_height:
        return True
    else:
        return False

def process_quick_read(boxes_np: np.ndarray, labels_np: np.ndarray) -> str:
    """
    Processes boxes and labels from a "quick read" scenario to extract the digits.
    Args:
        boxes_np: A NumPy array of shape (N, 4) containing bounding boxes
                  (ymin, xmin, ymax, xmax).
        labels_np: A NumPy array of shape (N,) containing corresponding integer labels.
    Returns:
        A string representing the extracted digits in their correct order.
    """
    box_centers: np.ndarray = (boxes_np[:, 1] + boxes_np[:, 3]) / 2
    boxes_labels: np.ndarray = np.concatenate((
        boxes_np, box_centers.reshape(-1, 1), labels_np.reshape(-1, 1)
    ), axis=1)

    # Sort top to bottom based on ymin (assuming index 1)
    boxes_labels = boxes_labels[boxes_labels[:, 4].argsort()]

    # Sort left to right within each group of 4 digits
    sorted_digits: np.ndarray = np.empty(NUM_QR_DIGS)
    for i in range(0, NUM_QR_DIGS, NUM_QR_GROUP):
        sorted_digits[i:i+NUM_QR_GROUP] = boxes_labels[i:i+NUM_QR_GROUP][
            boxes_labels[i:i+NUM_QR_GROUP, 0].argsort()
        ][:,-1]

    # Convert array to string and remove unnecessary characters
    return np.array2string(sorted_digits, separator='')[1:-1:2]

class OCR(nn.Module):
    # modified from: https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/ssd/ssd.py
    def __init__(self, num_classes: int, base_net: nn.ModuleList, source_layer_indexes: List[int],
                 classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList, is_test=False, config=None, device=None):
        super(OCR, self).__init__()

        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, Tuple):
                path = end_layer_index
                end_layer_index = end_layer_index[0]
            else:
                path = None
            for layer in self.base_net[start_layer_index: end_layer_index]:
                x = layer(x)
            y = x
            if path:
                sub = getattr(self.base_net[end_layer_index], path[1])
                for layer in sub[:path[2]]:
                    x = layer(x)
                y = x
                for layer in sub[path[2]:]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        
        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return confidence, location

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


class OCRPredictor:
    # modified from: https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/ssd/predictor.py
    def __init__(self, net, size, mean=0.0, std=1.0,
                 iou_threshold=0.45, filter_threshold=0.01, candidate_size=200, device=None):
        self.net = net
        self.transform = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net.to(self.device)
        self.net.eval()


    def predict(self, image, top_k=-1, prob_threshold=None):
        cpu_device = torch.device("cpu")
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            scores, boxes = self.net.forward(images)
        boxes = boxes[0]
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = utils.nms(box_probs,
                                      iou_threshold=self.iou_threshold,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]
