# Car-side Workspace Overview

- This workspace contains two ROS (catkin) packages:
  - `path_follower`: Mobile robot line-following controller consuming line detections and lidar safety.
  - `line_detector`: TensorRT-based line and intersection detector publishing as `path_follower/Lines`.

## Requirements
- ROS (catkin) and a supported distro (e.g., Noetic/Melodic)
- Python libs: `numpy`, `opencv-python`, `cv_bridge`
- Inference libs (for `line_detector`): `TensorRT`, `pycuda`
- Hardware: USB camera publishing `sensor_msgs/Image`; lidar publishing `sensor_msgs/LaserScan`; odometry publishing `geometry_msgs/PoseWithCovarianceStamped`; a base consuming `geometry_msgs/Twist`.

## Build
- From the workspace root:
  - `catkin_make`
  - `source devel/setup.bash`

## Quick Start
- Start detector:
  - `roslaunch line_detector line_detector.launch`
- Start follower:
  - `roslaunch path_follower line_follower.launch`

## Documentation
- `path_follower/README.md`: Parameters, topics, and control details
- `line_detector/README.md`: Detector configuration, topics, and dependencies