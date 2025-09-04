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

import numpy as np


import argparse
from filterpy.kalman import KalmanFilter
import torch

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
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    
    return (o)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] or [x1,y1,x2,y2,cartesian_x,cartesian_z] 
    and returns z in the form [x,y,s,r,cartesian_x,cartesian_z] where:
    - x,y is the centre of the box
    - s is the scale/area
    - r is the aspect ratio
    - cartesian_x,cartesian_z are real-world Cartesian coordinates
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    
    # Include Cartesian coordinates if provided
    if len(bbox) > 5:
        cartesian_x = bbox[4]
        cartesian_z = bbox[5]
        return np.array([x, y, s, r, cartesian_x, cartesian_z]).reshape((6, 1))
    else:
        # Default to 0 if not provided
        return np.array([x, y, s, r, 0, 0]).reshape((6, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r,cartesian_x,cartesian_z] and returns it in the form
      [x1,y1,x2,y2,score,cartesian_x,cartesian_z] where:
      - x1,y1 is the top left
      - x2,y2 is the bottom right
      - cartesian_x,cartesian_z are the real-world coordinates
    """

    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    cartesian_x = x[4] if len(x) > 4 else 0
    cartesian_z = x[5] if len(x) > 5 else 0
    
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., 
                          cartesian_x, cartesian_z]).reshape((1, 6))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., 
                          score, cartesian_x, cartesian_z]).reshape((1, 7))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=11, dim_z=6)
        self.kf.F = np.array([
        #   [u, v, s, r, x, z, uu,vv,ss,xx,zz]
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
        #   [u, v, s, r, x, z,uu,vv,ss,xx,zz]
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        ])

        self.kf.R[2:4, 2:4] *= 10. # This one has to fixed and shriknked to represent the certainty of the measurements
        self.kf.R[4:, 4:] *= 0.1

        self.kf.P[:, :4] *= 10.
        self.kf.P[:, 4:6] *= 0.01 # For cartesian coordinates
        self.kf.P[:, 6:] *= 1000.  # give high uncertainty to the unobservable initial velocities

        self.kf.Q[8, 8] *= 0.01
        self.kf.Q[:, 4:6] *= 0.2 # x, y
        self.kf.Q[:, 6:9] *= 0.01
        self.kf.Q[:, 9:] *= 0.25 #x_dot, _dot


        # Convert bbox to state vector [x,y,s,r,cartesian_x,cartesian_z]
        state = convert_bbox_to_z(bbox)
        self.kf.x[:state.shape[0]] = state
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        # Convert bbox to measurement vector and update
        measurement = convert_bbox_to_z(bbox)
        self.kf.update(measurement)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

def squared_mahalonobis_distance(detections, trackers):
    n_detections = detections.shape[0]
    n_trackers = len(trackers)

    distance_matrix = np.zeros((n_detections, n_trackers))

    for i in range(n_detections):

        z = detections[i, 4:].reshape(-1, 1)

        for j, tracker in enumerate(trackers):
            kf = tracker.kf

            y = z - kf.x[4:6]
            H = np.zeros((2, 11))
            H[0, 4] = 1
            H[1, 5] = 1
            S = kf.H[4:, :] @ kf.P @ kf.H[4:, :].T + kf.R[4:, 4:]

            # Mahalanobis
            S_inv = np.linalg.inv(S)

            # This is the correct Mahalanobis distance calculation
            distance = y.T @ S_inv @ y
            distance = float(distance)
            distance_matrix[i, j] = distance

    return distance_matrix

def associate_detections_to_trackers(detections, trackers, kfs, iou_threshold = 0.3, mb_threshold = 5.9915, use_mb = False):

  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = iou_batch(detections, trackers)\
  
#   print("IOU matrix")
#   print(iou_matrix)

  if use_mb:
    mb_matrix = squared_mahalonobis_distance(detections, kfs)
    # Add the Mahalanobies Distances in Here
    # print("MB matrix")
    # print(mb_matrix)
  else:
    mb_matrix = iou_matrix


  # I need to play around in here to include the MB
  ###################################################
  if min(iou_matrix.shape) > 0 and min(mb_matrix.shape) > 0:
    if use_mb:
        gate_mask = (iou_matrix > iou_threshold) & (mb_matrix < mb_threshold)
    else:
        gate_mask = (iou_matrix > iou_threshold)

    a = gate_mask.astype(np.int32)
    # print("gate_mask")
    # print(gate_mask)

    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    # elif a.sum(1).max() == 0 and a.sum(0).max() == 0:
    #     matched_indices = np.empty(shape=(0,2))
    else:
        if use_mb:
            matched_indices = linear_assignment(-iou_matrix + mb_matrix)
        else:
            matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))
  # ################################################

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  # filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3, mb_threshold = 5.9915, use_mb = False):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.mb_threshold = mb_threshold
        self.trackers = []
        self.frame_count = 0
        self.use_mb = use_mb
        self.detections_imgs = []
        self.detection_kpts = []
        self.original_kpts = []
        KalmanBoxTracker.count = 0

    def reset_counter(self):
        KalmanBoxTracker.count = 0

    def update(self, dets, detections_imgs, detection_kpts, original_kpts):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,cartesian_x,cartesian_z],...] 
                 where cartesian_x and cartesian_z are real-world coordinates
        Requires: this method must be called once for each frame even with empty detections 
                 (use np.empty((0, 7)) for frames without detections).
        Returns a similar array, where the format is [[x1,y1,x2,y2,cartesian_x,cartesian_z, id],...] 
                with ID being the object identifier.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 6))  # Updated to include cartesian_x and cartesian_z
        to_del = []
        ret = []
        ret_detections_imgs = []
        ret_detection_kpts = []
        ret_original_kpts = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            self.detections_imgs.pop(t)
            self.detection_kpts.pop(t)
            self.original_kpts.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.trackers, self.iou_threshold, self.mb_threshold, use_mb=self.use_mb)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            self.detections_imgs[m[1]] = detections_imgs[m[0]] 
            self.detection_kpts[m[1]] = detection_kpts[m[0]]
            self.original_kpts[m[1]] = original_kpts[m[0]]


        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

            self.detections_imgs.append(detections_imgs[i])
            self.detection_kpts.append(detection_kpts[i])
            self.original_kpts.append(original_kpts[i])

        i = len(self.trackers)
        for trk, det_imgs, det_kpts, det_orig_kpts in zip(reversed(self.trackers), reversed(self.detections_imgs), reversed(self.detection_kpts), reversed(self.original_kpts)):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # Include cartesian coordinates in the return value
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
                ret_detections_imgs.append(det_imgs)
                ret_detection_kpts.append(det_kpts)
                ret_original_kpts.append(det_orig_kpts)

            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                self.detections_imgs.pop(i)
                self.detection_kpts.pop(i)
                self.original_kpts.pop(i)
                 
        if (len(ret) > 0):
            return (np.concatenate(ret), torch.stack(ret_detections_imgs, dim=0), torch.stack(ret_detection_kpts, dim=0), np.stack(ret_original_kpts, axis=0))
        return None # Updated to include cartesian_x and cartesian_z in output

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args