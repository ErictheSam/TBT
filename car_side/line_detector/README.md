# Line Detector

## Dependencies & Environment
- ROS (catkin): `rospy`, `sensor_msgs`, `std_msgs`, `geometry_msgs`
- Python libs: `numpy`, `opencv-python`, `torch`
- Inference libs: `TensorRT`, `pycuda`
- Model: TensorRT engine file (default `models/detector.trt` or script-configured `.trt`)

## Launch
- Command: `roslaunch line_detector line_detector.launch`
- Includes:
  - Node `infer` (script `scripts/predict.py`)
  - USB camera bringup: `$(find usb_cam)/launch/usb_cam-line-follower.launch`

## Parameters (line_detector.launch)
- `image_size` (`int`): preprocessing image size (default `280`)
- `usbcam_height` (`int`): camera input height (default `400`)
- `usbcam_width` (`int`): camera input width (default `640`)
- `model_name` (`string`): TensorRT engine filename under `models/` (default `detector.trt` as the pretrained tensorrt model)

## Topics
- Input:
  - `/usb_cam/image_raw` (`sensor_msgs/Image`)
- Output:
  - `/output/lines_and_intersecs` (`path_follower/Lines`)
    - `lines`: `std_msgs/Float32MultiArray[]` with `[x_start, y_start, x_end, y_end]`
    - `points`: `std_msgs/Float32MultiArray[]` with `[x, y]`

## Build (CMake/catkin)
- Package depends on: `rospy`, `sensor_msgs`, `std_msgs`, `geometry_msgs`
- Steps:
  - `catkin_make`
  - `source devel/setup.bash`

## Directory Layout
- `scripts/`: inference entry `predict.py`; helpers `inference.py`, `engine_utils.py`, `utils.py`
- `launch/`: `line_detector.launch`
- `models/`: TensorRT engine files
- `package.xml`, `CMakeLists.txt`

## Quick Start
- Model preparation: 
  - Place the TensorRT engine file in `models/` directory.
- Start detector: `roslaunch line_detector line_detector.launch`