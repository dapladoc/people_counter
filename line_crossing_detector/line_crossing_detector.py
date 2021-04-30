# flake8: noqa: F821
from collections import namedtuple
from typing import List

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from deep_sort import nn_matching, preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from line_crossing_detector.utils import LimitedSizeDict
from tools import generate_detections
from yolo4.yolo import YOLO

LineCrossingDetection = namedtuple("LineCrossingDetection", ("track_id", "line_id"))


physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


class LineCrossingDetector:
    """This class implements line crossing detector logic.
    Parameters:
        config: OmegaConf instance with configuration of the detector.
    """

    def __init__(self, config):
        self.config = config
        self.lines = self._create_lines(config.lines)
        self.human_detector = YOLO(**config.human_detector)
        self.human_class_name = "person"
        self.tracker = self._create_tracker(config.tracker)
        self.features_encoder = generate_detections.create_box_encoder(config.tracker.deepsort_model_path, batch_size=1)
        self.previous_tracks_positions = LimitedSizeDict(size_limit=config.lines_crossing_detector.history_size)

    @staticmethod
    def _create_lines(config: "omegaconf.DictConfig") -> List["Line"]:
        """Read lines settings from config and return Lines instances."""
        lines = []
        for line in config:
            point1 = (line.begin.x, line.begin.y)
            point2 = (line.end.x, line.end.y)
            lines.append(Line(point1, point2, line.id))
        return lines

    @staticmethod
    def _create_tracker(config: "omegaconf.DictConfig") -> Tracker:
        """Create Tracker instance according to the config."""
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", config.max_cosine_distance, config.nn_budget)
        return Tracker(metric)

    def detect(self, frame: np.ndarray, visualize: bool = False) -> List[LineCrossingDetection]:
        """Run full detection cycle: detect humans, run tracker, inspect whether there are tracks that
        crossed the lines, and return list of LineCrossingDetections as a result."""
        human_detections = self._run_human_detector_and_encoder(frame)
        human_detections = self._run_nms(human_detections)
        self._update_tracker(human_detections)
        line_crossing_detections = []
        for track in self.tracker.tracks:
            current_position = track.get_center()
            if track.track_id not in self.previous_tracks_positions:
                self.previous_tracks_positions[track.track_id] = current_position
                continue
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            previous_position = self.previous_tracks_positions[track.track_id]
            for line in self.lines:
                if line.is_cross(current_position, previous_position):
                    line_crossing_detections.append(LineCrossingDetection(track.track_id, line.id))
            self.previous_tracks_positions[track.track_id] = current_position
        if visualize:
            self._visualize(frame, line_crossing_detections, human_detections)
        return line_crossing_detections

    def _run_human_detector_and_encoder(self, frame: np.ndarray) -> List[Detection]:
        """Run human detector and evaluate humans emdeddings."""
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxes, confidence, classes = self.human_detector.detect_image(image)
        features = self.features_encoder(frame, boxes)
        detections = [
            Detection(bbox, confidence, cls, feature)
            for bbox, confidence, cls, feature in zip(boxes, confidence, classes, features)
            if cls == self.human_class_name
        ]
        return detections

    def _run_nms(self, human_detections: List[Detection]) -> List[Detection]:
        """Run non maximum supression."""
        boxes = np.array([d.tlwh for d in human_detections])
        scores = np.array([d.confidence for d in human_detections])
        indices = preprocessing.non_max_suppression(boxes, self.config.nms.max_overlap, scores)
        detections = [human_detections[i] for i in indices]
        return detections

    def _update_tracker(self, detections: List[Detection]):
        """Run tracker for the list of detections."""
        self.tracker.predict()
        self.tracker.update(detections)

    def _visualize(
        self, frame: np.ndarray, crossing_detections: List[LineCrossingDetection], human_detections: List[Detection]
    ):
        """Draw lines and bounding boxes on the frame."""

        def get_random_color():
            return tuple(np.random.randint(0, 256) for _ in range(3))

        if getattr(self, "_lines_colors", None) is None:
            self._lines_colors = {line.id: get_random_color() for line in self.lines}
        if getattr(self, "_tracks_colors", None) is None:
            self._tracks_colors = dict()
        for line in self.lines:
            line.draw(frame, self._lines_colors[line.id], 2)
        for track in self.tracker.tracks:
            if track.track_id not in self._tracks_colors:
                self._tracks_colors[track.track_id] = (255, 255, 255)
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            for d in crossing_detections:
                if d.track_id == track.track_id:
                    self._tracks_colors[track.track_id] = self._lines_colors[d.line_id]
            bbox = track.to_tlbr()
            text_color = (0, 255, 0)
            text_size = 1.5e-3 * frame.shape[0]
            text = f"ID: {track.track_id}"
            text_coords = (int(bbox[0]), int(bbox[1]))
            center = track.get_center()
            rect_color = self._tracks_colors[track.track_id]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), rect_color, 2)
            cv2.putText(frame, text, text_coords, 0, text_size, text_color, 1)
            cv2.circle(frame, (int(center[0]), int(center[1])), 5, rect_color, 2)
        for d in human_detections:
            bbox = d.to_tlbr()
            score = "%.2f" % round(d.confidence * 100, 2) + "%"
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cls = d.cls
            cv2.putText(
                frame, str(cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0, 1.5e-3 * frame.shape[0], (0, 255, 0), 1
            )


class Line:
    def __init__(self, point1, point2, id_):
        """The line in 2D could be defined as Ax + By + C = 0.
        A, B, C could be defined from the equation
        |x - x1      y - y1|
        |                  | = 0
        |x2 - x1    y2 - y1|
        Parameters:
        point1: (x1, y1) of the beginning of the line
        point2: (x2, y2) of the end of the line
        """
        x1, y1 = point1
        x2, y2 = point2
        self.begin = point1
        self.end = point2
        self.a = y2 - y1
        self.b = -(x2 - x1)
        self.c = -x1 * (y2 - y1) + y1 * (x2 - x1)
        self.id = id_

    def signed_distance(self, point):
        """Return signed distance from point to line
        Parameters:
        point: (x, y)
        """
        x, y = point
        return self.a * x + self.b * y + self.c

    def is_cross(self, point1, point2):
        """Return true if segment (point1, point2) crosses the line.
        Parameters:
            point1: (x, y) of the beginning of the segment
            point2: (x, y) of the end of the segment
        """
        return self.signed_distance(point1) * self.signed_distance(point2) < 0

    def draw(self, frame, color, thickness):
        return cv2.line(frame, self.begin, self.end, color, thickness)
