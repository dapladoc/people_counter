# Introduction
This project is completely based on:
https://github.com/LeonLok/Deep-SORT-YOLOv4

The code was slightly modified by using `black` and `isort`. There were added classes and functions to count people
by generating lines crossing events.

The main class is `LineCrossingDetector` that does all the job: detects persons,
track them, and detect the lines crossing by the detected persons. Method `detect` of this class returns list of matches
`(track_id, line_id)`.

The `demo.py` was modified to show how it's work.

All the settings were moved to `config.yaml` file.

# YOLO v4 model
[Download](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT) and convert the Darknet YOLO v4 model
to a Keras model by modifying `convert.py` accordingly and run:
```
python convert.py
```
Then run demo.py:
```
python demo.py
```

# Dependencies
```
imutils==0.5.3
keras==2.3.1
matplotlib==3.2.1
numpy==1.18.4
opencv-python==4.2.0.34
pillow==7.1.2
scikit-learn==0.23.1
scipy==1.4.1
tensorflow-gpu==2.2.0
omegaconf==2.0.6
```
