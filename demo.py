#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import warnings

import cv2
import tensorflow as tf
from omegaconf import OmegaConf
from tqdm import tqdm

from line_crossing_detector.line_crossing_detector import LineCrossingDetector, PeopleCounter
from videocaptureasync import VideoCaptureAsync

# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
warnings.filterwarnings("ignore")


def main():
    input_video_filepath = sys.argv[1]
    write_video_flag = False
    output_video_filepath = ""
    if len(sys.argv) > 2:
        output_video_filepath = sys.argv[2]
        write_video_flag = True
    async_video_flag = False

    config = OmegaConf.load("config.yaml")
    detector = LineCrossingDetector(config)
    counters = [PeopleCounter(**c) for c in config.people_counter]

    if async_video_flag:
        video_capture = VideoCaptureAsync(input_video_filepath)
    else:
        video_capture = cv2.VideoCapture(input_video_filepath)

    if async_video_flag:
        video_capture.start()

    if write_video_flag:
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        output_writer = cv2.VideoWriter(output_video_filepath, fourcc, 30, (w, h))

    frame_index = 0
    pbar = tqdm(total=int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    while True and frame_index < 12000:
        ret, frame = video_capture.read()
        frame_index = frame_index + 1
        if not ret:
            break
        if frame_index < 10000:
            continue

        detections = detector.detect(frame, visualize=True)
        for counter in counters:
            counter.update(detections, frame_index)
            counter.visualize(frame)
        for d in detections:
            print(f"Frame: {frame_index}. Track id: {d.track_id}. Line id: {d.line_id}")
        if write_video_flag:
            output_writer.write(frame)
        pbar.update()

    if async_video_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if write_video_flag:
        output_writer.release()


if __name__ == "__main__":
    main()
