## Running OCR inference

### System Requirements

This code was tested with Ubuntu(20.04.3 LTS) with Python 3.8.10, MacOSX Sonoma 14.2.1 with Python 3.9.16

#### Setup Virtual environment and install requirements

```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements
```
#### To run the demo with a predefined video input, use the following
Check out the `test_videos` directory for sample videos.

```
python3 run_ocr_demo.py --video_file ./test_videos/amex.mp4
```
#### To run the demo on the frames from your webcam, use the following
Place your card infront of the webcam.

```
python3 run_ocr_demo.py
```

## For more information on the project
Read the full paper [Doing good by fighting fraud: Ethical anti-fraud
systems for mobile payments](https://arxiv.org/pdf/2106.14861.pdf)

## Acknowledgement
We would like to thank [Max deGroot](https://github.com/amdegroot) for [ssd.ptorch](https://github.com/amdegroot/ssd.pytorch) and 
[Hao](https://github.com/qfgaohao) for [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd) and [Joseph Redmon](https://github.com/pjreddie)
for [YOLO and Darknet](https://github.com/pjreddie/darknet). We borrow from these projects.
