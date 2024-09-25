"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function
# import matplotlib
# matplotlib.use('TkAgg')
import os
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    Computes IOU between two sets of bounding boxes in the form [x1, y1, x2, y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)  # Shape: (1, N, 4)
    bb_test = np.expand_dims(bb_test, 1)  # Shape: (M, 1, 4)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1, y1, x2, y2] and returns z in the form
    [x, y, s, r] where x, y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h    # scale is just area
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x, y, s, r] and returns it in the form
    [x1, y1, x2, y2] where x1, y1 is the top left and x2, y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / (w + 1e-6)
    if(score is None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, class_id):
        """
        Initialises a tracker using initial bounding box and class ID.
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.  # Give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.1
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # Store class ID
        self.class_id = class_id

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers_multiclass(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked objects (both represented as bounding boxes with class IDs).

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers.
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

    # Split detections and trackers by class
    dets_classes = detections[:, -1]
    trks_classes = trackers[:, -1]

    matches = []
    unmatched_detections = []
    unmatched_trackers = []

    for cls in np.unique(np.concatenate((dets_classes, trks_classes))):
        det_indices = np.where(dets_classes == cls)[0]
        trk_indices = np.where(trks_classes == cls)[0]

        if len(det_indices) == 0:
            unmatched_trackers.extend(trk_indices)
            continue
        if len(trk_indices) == 0:
            unmatched_detections.extend(det_indices)
            continue

        cls_dets = detections[det_indices, :4]
        cls_trks = trackers[trk_indices, :4]

        iou_matrix = iou_batch(cls_dets, cls_trks)

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                cls_matched_indices = np.stack(np.where(a), axis=1)
            else:
                cls_matched_indices = linear_assignment(-iou_matrix)
        else:
            cls_matched_indices = np.empty(shape=(0, 2))

        for m in cls_matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(det_indices[m[0]])
                unmatched_trackers.append(trk_indices[m[1]])
            else:
                matches.append([det_indices[m[0]], trk_indices[m[1]]])

        unmatched_det_indices = [d for d in det_indices if d not in [det_indices[m[0]] for m in cls_matched_indices]]
        unmatched_trk_indices = [t for t in trk_indices if t not in [trk_indices[m[1]] for m in cls_matched_indices]]

        unmatched_detections.extend(unmatched_det_indices)
        unmatched_trackers.extend(unmatched_trk_indices)

    matches = np.array(matches)
    unmatched_detections = np.array(unmatched_detections)
    unmatched_trackers = np.array(unmatched_trackers)

    return matches, unmatched_detections, unmatched_trackers


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 6))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1, y1, x2, y2, score, class_id], ...]
        Requires: this method must be called once for each frame even with empty detections
        (use np.empty((0, 6)) for frames without detections).
        Returns:
          an array similar to the input, but with the last column being the object ID and class ID.
        """
        self.frame_count += 1

        # Predict new locations of all trackers
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0, trk.class_id]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers_multiclass(dets, trks, self.iou_threshold)

        # Update matched trackers with assigned detections
        for m in matched:
            det_idx, trk_idx = m[0], m[1]
            self.trackers[trk_idx].update(dets[det_idx, :4])
            # Update class_id (if necessary)
            self.trackers[trk_idx].class_id = dets[det_idx, -1]

        # Create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4], class_id=dets[i, -1])
            self.trackers.append(trk)

        # Remove dead tracklets
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1, trk.class_id])).reshape(1, -1))
            i -= 1
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)

        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 6))

