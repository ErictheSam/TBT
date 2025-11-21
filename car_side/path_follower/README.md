# Path Follower

## Dependencies & Environment
- ROS (catkin), recommended Noetic/Melodic
- Mobile base bringup capable of consuming operations `geometry_msgs/Twist` and omitting odometry `/geometry_msgs/PoseWithCovarianceStamped` (example: `turn_on_wheeltec_robot`)
- Laser scanner publishing `sensor_msgs/LaserScan` on topic `scan` (example: `wheeltec_lidar.launch`)
- Custom message `simple_follower/Lines`, Python libs `cv_bridge`, `numpy`

## Launch
- Command: `roslaunch path_follower path_follower.launch`
- Dependencies:
  - Image prediction from `line_detector` 
  - Base and lidar bringup (examples under `turn_on_wheeltec_robot/*`)
- Control output: `line_follow.py` publishes `geometry_msgs/Twist` to `/cmd_vel`.

## Parameters (path_follower.launch)
- `max_speed`: maximum cruising linear speed (m/s)
- `smooth_speed`: linear speed used for soft start/stop (m/s)
- `stop_speed`: slow linear speed near intersections (m/s)
- `rotate_speed`: maximum rotational speed (rad/s)
- `perception_field`: the field of the top-down camera (m/s)
- `odometry_delta`: the delta threshold for wheel odometry calibration (m)
- `image_size`: the size of image preprocessed by line detector (default `280`)
- `window_size`: lidar sliding-window size for neighbor consistency check
- `distance_tolerance`: lidar distance tolerance for noise rejection (m)
- `trip`: workflow ID name (selects the `ID_*` namespace)
- `PID_controller`: loads 2D PID params from `parameters/PID_line_following_param.yaml`
- `ID_go`: loads workflow config from the example`parameters/ID_go.yaml` (`max_cross`, `next_status`, `skip_dir`, `length_dir`)

## Yaml Example Format
This yaml contains the topological map of a **point-to-point** line-following task.
- `max_cross`: maximum number of line crossings before changing status
- `next_status`: next status to transition to after each `max_cross` crossings with status numbers as follows
  - -1: terminate
  - 0: move straight
  - 1: move under smooth_speed
  - 2: move under stop_speed
  - 10: rotate left
  - 11: rotate right
- `skip_dir`: the ranks of contaminated intersections to skip and the expected distance (m/s) to skip
- `length_dir`: the length of a single tile (m) on each status 0 (a new straight section)

## Build (CMake/catkin)
- Dependencies: `roscpp`, `rospy`, `std_msgs`, `sensor_msgs`, `geometry_msgs`, `message_generation`, `message_runtime`
- Generated messages: `msg/Lines.msg`
- Build steps:
  - `catkin_make`
  - `source devel/setup.bash`

## Python Requirements
- `numpy`
- `cv_bridge` (install via ROS: `sudo apt install ros-<distro>-cv-bridge`)
- `rospy` (provided by ROS)

## Topics
- Input:
  - `/output/lines_and_intersecs` (`msg/Lines`)
  - `/robot_pose_ekf/odom_combined` (odometry/fused pose)
  - `scan` (lidar)
- Output:
  - `/cmd_vel` (`geometry_msgs/Twist`)


## Quick Start
- Update `launch/line_follower.launch` with your own parameters, and change the base and lidar settings to your own in lines 17-18.
- Start line follower:
  - `roslaunch simple_follower line_follower.launch trip_id:=ID_go`

## Notes
- Ensure your line detector publishes `simple_follower/Lines` to `/output/lines_and_intersecs`