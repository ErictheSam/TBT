#!/usr/bin/env python3

from controllers import SimplePID
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from lidar_safety import LidarSafety
from path_follower.msg import Lines
from sensor_msgs.msg import LaserScan

import cv_bridge
import numpy as np
import rospy
import time

class LineFollower:
    """
    LineFollower is a ROS node that enables a robot to follow a line on the ground
    using camera input, laser-scan safety checks, and odometry feedback.

    It handles:
    - Line detection and tracking via /output/lines_and_intersecs
    - Obstacle avoidance using LidarSafety
    - Speed control based on path status
    - Rotation alignment at intersections
    - Odometry-based progress tracking
    """
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.window_size = rospy.get_param('~window_size', rospy.get_param('~winSize'))
        self.distance_tolerance = rospy.get_param(
            '~distance_tolerance', rospy.get_param('~deltaDist')
        )
        self.cmd = Twist()
        self.image_size = rospy.get_param('~image_size')
        max_speed = rospy.get_param('~max_speed')
        smooth_speed = rospy.get_param('~smooth_speed')
        stop_speed = rospy.get_param('~stop_speed')
        self.speed_levels = [max_speed, smooth_speed, stop_speed]
        self.rotate_speed = rospy.get_param('~rotate_speed')
        self.perception_field = rospy.get_param('~perception_field')
        self.odom_delta = rospy.get_param('~odometry_delta')
        self.lidar_angle_l = 80.0
        self.lidar_angle_r = 80.0
        self.stopped = False
        self.stop_past = True
        pid_param = rospy.get_param('~PID_controller')
        trip_gen = rospy.get_param('~trip')
        trip_id = rospy.get_param('~' + trip_gen)
        self.pid_controller = SimplePID([0.0,0.0], pid_param['P'], pid_param['I'], pid_param['D'])
        self.pasts = 0
        self.frame_id = 0
        self.max_cross = trip_id['max_cross']
        self.cross_point = 0
        self.next_status = trip_id['next_status']
        self.skip_dir = trip_id['skip_dir']
        self.length_dir = trip_id['length_dir']
        self.status = self.next_status[0]
        self.status_point = 1
        self.delta = 0
        self.last_xyz = [0,0,0]
        self.last_x = -1
        self.last_y = -1
        self.last_stop_time = time.time()
        self.cur_xyz = [0,0,0]
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.rotating = 0
        self.gap_point = 0
        self.mistake_point = 0
        self.delta = 0.0
        self.line_sub = rospy.Subscriber("/output/lines_and_intersecs", Lines, self.image_callback)
        self.odom_sub = rospy.Subscriber(
            "/robot_pose_ekf/odom_combined",
            PoseWithCovarianceStamped,
            self.odom_callback,
        )
        self.safety = LidarSafety(
            self.window_size,
            self.distance_tolerance,
            self.lidar_angle_l,
            self.lidar_angle_r
        )
        self.scan_subscriber = rospy.Subscriber('scan', LaserScan, self.avoid_distance_callback)


    def odom_callback(self, msg):
        """
        Callback for odometry updates from /robot_pose_ekf/odom_combined.
        Stores the current robot position (x, y, z) in self.cur_xyz.
        """
        self.cur_xyz[0] = msg.pose.pose.position.x
        self.cur_xyz[1] = msg.pose.pose.position.y
        self.cur_xyz[2] = msg.pose.pose.position.z

    def rotate_to_align(self, lines, points, clockwise):
        """
        Rotate the robot to align with the dominant line at an intersection.

        Uses the two detected lines to compute their angles relative to the vertical
        (90°).  When either line is close to vertical, rotation stops.  Otherwise,
        the robot spins in the requested direction with a speed proportional to the
        remaining mis-alignment.

        Args:
            lines (list): Two line segments, each as [x1, y1, x2, y2].
            points (list): Intersection point(s) as [[x, y]] (unused in rotation logic).
            clockwise (bool): True → rotate clockwise, False → counter-clockwise.

        Returns:
            bool: True if alignment is achieved, False otherwise.
        """
        self.cmd.linear.x = 0
        self.cmd.linear.y = 0
        self.last_xyz[0] = self.cur_xyz[0]
        self.last_xyz[1] = self.cur_xyz[1]
        self.last_xyz[2] = self.cur_xyz[2]
        if len(lines) != 2:
            self.cmd.angular.z = self.cmd.angular.z / 2
            return False
        l1 = lines[0]
        py = points[0][1]
        px = points[0][0]
        self.last_y = py
        self.last_x = px
        angle_l1 = np.arctan2(l1[0]-l1[2],l1[1]-l1[3]) * 180 / np.pi
        if angle_l1 < 0.0:
            angle_l1 += 180
        l2 = lines[1]
        angle_l2 = np.arctan2(l2[0]-l2[2],l2[1]-l2[3]) * 180/ np.pi
        if angle_l2 < 0.0:
            angle_l2 += 180
        if abs(angle_l1-90) < 20 or abs(angle_l2-90) < 20:
            if self.rotating > 3:
                self.rotating = 0
                self.cmd.angular.z = 0
                return True
        else:
            self.rotating += 1

        # Calculate the to-rotate angular and the speed
        ang = min(abs(angle_l1-90),abs(angle_l2-90)) * np.pi/180
        if self.rotating > 0:
            speed = min(self.rotate_speed, ang)
        else:
            speed = self.rotate_speed
        if clockwise:
            self.cmd.angular.z = speed
        else:
            self.cmd.angular.z = -speed
        return False

    def avoid_distance_callback(self, scan_data):
        """
        Callback for incoming LaserScan messages.

        Delegates safety evaluation to the LidarSafety module and updates
        the robot's stopped state, stop-hold flag, and last-stop timestamp.
        """
        res = self.safety.update(scan_data, self.status, self.speed_levels)
        self.stopped = res['stopped']
        self.stop_past = res['stop_hold']
        self.last_stop_time = res['last_stop_time']

    def image_callback(self, msg):
        """
        Callback for incoming line-and-intersection data from /output/lines_and_intersecs.

        Processes detected line segments and intersection points to:
        - Decide when to rotate at intersections (status 10/11)
        - Compute lateral deviation (d_len) and heading error (theta) for PID control
        - Track crossed tiles via odometry and trigger speed changes or stops
        - Publish final Twist command to /cmd_vel
        """
        line_set = msg.lines
        point_set = msg.points

        line_set = [x.data for x in line_set]
        point_set = [x.data for x in point_set]
        self.frame_id += 1
        if self.status <= -1:
            self.delta = 0.0
            self.cmd.linear.x = 0.0
            self.cmd.linear.y = 0.0
            self.cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(self.cmd)
            return
        elif self.status == 10 or self.status == 11:
            self.delta = 0.0
            counter = self.status % 2 == 0
            stop = self.rotate_to_align(line_set, point_set, counter)
            if stop:
                self.pasts = 0
                self.stop_past = True
                self.last_stop_time = time.time()
                self.status = self.next_status[self.status_point]
                # Update the tile length when a new line is established
                if self.status == 0:
                    self.gap_point += 1
                self.status_point += 1
            self.cmd_vel_pub.publish(self.cmd)
            return
        else:
            d_len = 0
            theta = 0
            if len(line_set) > 2:
                return
            # Intersection
            if len(line_set) == 2:
                l1 = line_set[0]
                angle_l1 = np.arctan2(l1[0]-l1[2], l1[1]-l1[3]) * 180/np.pi
                if angle_l1 < 0.0:
                    angle_l1 += 180
                l2 = line_set[1]
                angle_l2 = np.arctan2(l2[0]-l2[2], l2[1]-l2[3]) * 180/np.pi
                if angle_l2 < 0.0:
                    angle_l2 += 180
                ts = 0
                if angle_l1 < 30:
                    theta = angle_l1 * np.pi/180
                elif angle_l1 > 150:
                    theta = (angle_l1-180) * np.pi /180
                elif angle_l2 < 30:
                    theta = angle_l2 * np.pi/180
                    ts = 1
                elif angle_l2 > 150:
                    theta = (angle_l2-180) * np.pi/180
                    ts = 1
                else:
                    return
                mid_line = line_set[ts]
                d_len = (
                    mid_line[2]
                    + (self.image_size/2 - mid_line[3])
                      * (mid_line[2] - mid_line[0])
                      / (mid_line[3] - mid_line[1])
                    - self.image_size/2
                )
                if len(point_set) != 0:
                    # One key frame
                    x , y = point_set[0][0], point_set[0][1]

                    # Calculate the y-projected distance between two key frames
                    angle_diff = abs(angle_l1 - angle_l2)
                    if (abs(angle_diff - 90) < 20 and
                        0 < x < self.image_size and
                        0 < y < self.image_size):
                        dist_change = np.sqrt(
                            (self.last_xyz[0] - self.cur_xyz[0]) ** 2
                            + (self.last_xyz[1] - self.cur_xyz[1]) ** 2
                            - (
                                self.perception_field
                                * (self.last_x - x)
                                / self.image_size
                            ) ** 2
                        )
                        dist_change += (
                            self.perception_field * (self.last_y - y) / self.image_size
                        )
                        special_skip = False
                        if not str(self.pasts) in self.skip_dir[self.cross_point]:
                            if (
                                min(dist_change, dist_change - self.delta)
                                > max(
                                    self.length_dir[self.gap_point] - self.odom_delta,
                                    self.odom_delta * 1.5,
                                )
                            ):
                                special_skip = True
                                self.delta = self.length_dir[self.gap_point] - dist_change

                                # Calculate the forgotten points between two frames
                                extra_dist = (
                                    dist_change
                                    - 2 * self.length_dir[self.gap_point]
                                    + self.perception_field
                                )
                                if (time.time() - self.last_stop_time) >= 1:
                                    self.stop_past = False
                                self.pasts += 1
                                if extra_dist > 0:
                                    miss_count = 1 + int(
                                        extra_dist // self.length_dir[self.gap_point]
                                    )
                                    self.mistake_point += miss_count
                                    self.pasts += miss_count
                        elif dist_change > self.skip_dir[self.cross_point][str(self.pasts)]:
                            extra_dist = (
                                dist_change
                                + self.perception_field
                                - self.skip_dir[self.cross_point][str(self.pasts)]
                                - self.length_dir[self.gap_point]
                            )
                            self.delta = (
                                self.skip_dir[self.cross_point][str(self.pasts)]
                                - dist_change
                            )
                            special_skip = True
                            if (time.time() - self.last_stop_time) >= 1:
                                self.stop_past = False
                            self.pasts += 2
                            if extra_dist > 0:
                                miss_count = 1 + int(extra_dist // self.length_dir[self.gap_point])
                                self.mistake_point += miss_count
                                self.pasts += miss_count
                        if special_skip or (dist_change < 0.05):
                            self.last_xyz[0] = self.cur_xyz[0]
                            self.last_xyz[1] = self.cur_xyz[1]
                            self.last_xyz[2] = self.cur_xyz[2]
                            self.last_y = y
                            self.last_x = x

                        # Approaching strategy: status 1 = slow down, status 2 = going to stop
                        if (self.length_dir[self.gap_point] <= 0.6 and
                            self.pasts + 2 == self.max_cross[self.cross_point]):
                            self.status = 1
                        elif self.pasts + 1 == self.max_cross[self.cross_point] \
                            or (self.pasts + 2 == self.max_cross[self.cross_point]
                                and str(self.pasts) in self.skip_dir[self.cross_point]):
                            self.status = 2
                        elif self.pasts == self.max_cross[self.cross_point]:
                            self.status = self.next_status[self.status_point]
                            self.status_point += 1
                            self.cmd.linear.x = 0.0
                            self.cmd.linear.y = 0.0
                            self.cmd.angular.z = 0.0
                            self.cmd_vel_pub.publish(self.cmd)
                            self.cross_point += 1
                            return
                        
            # One-way path      
            elif len(line_set) == 1:
                l1 = line_set[0]
                mid_line = l1
                angle_l1 = np.arctan2(l1[0]-l1[2],l1[1]-l1[3]) * 180 / np.pi
                if angle_l1 < 0:
                    angle_l1 += 180
                if angle_l1 < 90:
                    theta = angle_l1 * np.pi/180
                else:
                    theta = (angle_l1-180) * np.pi/180
                if abs(mid_line[3] - mid_line[1]) < abs(mid_line[2]-mid_line[0]):
                    d_len = 0
                else:
                    d_len = (
                        mid_line[2]
                        + (self.image_size/2 - mid_line[3])
                          * (mid_line[2] - mid_line[0])
                          / (mid_line[3] - mid_line[1])
                        - self.image_size/2
                    )
            else:
                self.cmd.linear.x = self.cmd.linear.x/2
                self.cmd.linear.y = self.cmd.linear.y/2
                self.cmd.angular.z = self.cmd.angular.z/2
                self.cmd_vel_pub.publish(self.cmd)
                return
           
            # Update the new command
            angular_speed = 0.0
            linear_speed = 0.0

            # When theta is too large or deviation is too far, stop the car and update
            if abs(theta) > 0.3 or abs(d_len) > self.image_size/3:
                self.cmd.linear.x = 0.0
            else:
                if self.stop_past:
                    speed_rank = max(self.status, 1)
                else:
                    speed_rank = self.status
                # Translatory velocity to recenter the car
                self.cmd.linear.x = self.speed_levels[speed_rank] * np.exp(-abs(2*theta))
            linear_speed, angular_speed = self.pid_controller.update([d_len, theta])
            self.cmd.linear.y = -linear_speed + np.tan(theta) * self.cmd.linear.x

            # Angular velocity to recenter the direction
            if abs(theta) < 0.02:
                self.cmd.angular.z = 0
            else:
                self.cmd.angular.z = 0.25 * angular_speed
            if self.stopped:
                self.cmd.linear.x = 0.0
                self.cmd.linear.y = 0.0
                self.cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(self.cmd)

rospy.init_node("line_follower")
follower = LineFollower()
rospy.spin()
