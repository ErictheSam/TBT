#!/usr/bin/env python3

import math
import numpy as np
import time

class LidarSafety:
    def __init__(self, window_size, distance_tolerance,
                 angle_left=80.0, angle_right=80.0):
        self.window_size = window_size
        self.distance_tolerance = distance_tolerance
        self.angle_left = angle_left
        self.angle_right = angle_right
        self.last_scan = None
        self.distance_avoidance = 0
        self.stopped = False
        self.stop_hold = True
        self.last_stop_time = time.time()

    def update(self, scan_data, state, speed_levels):
        """
        Process a new LiDAR scan and determine safety-related flags.

        Parameters
        ----------
        scan_data : sensor_msgs.msg.LaserScan
            The incoming LiDAR scan message containing ranges, angle limits,
            and angle increment.
        state : int
            Current navigation state used to select an appropriate stopping
            threshold (e.g., 0 for normal driving, 10/11 for special cases).
        speed_levels : list or tuple
            Speed-related parameters; the first element is used to scale the
            stopping threshold when state == 0.

        Returns
        -------
        dict
            {
                'min_distance': float,   # Closest valid obstacle distance
                'stopped': bool,         # True if the robot must remain stopped
                'stop_hold': bool,       # True if stopping logic is latched
                'last_stop_time': float  # Timestamp of the most recent stop
            }
        """
        ranges = np.array(scan_data.ranges)
        ang_min = scan_data.angle_min
        ang_max = scan_data.angle_max
        inc_deg = scan_data.angle_increment * 180 / math.pi
        right_max = int(self.angle_left / inc_deg)
        left_max = int(self.angle_right / inc_deg)
        left_right_inc = len(ranges) - right_max - left_max
        ranges_legal = np.concatenate(
            (ranges[0:right_max], ranges[-left_max:]), axis=0
        )
        ranges_legal_ang = np.concatenate(
            (
                [ang_min + scan_data.angle_increment * i
                 for i in range(right_max)],
                [ang_max - scan_data.angle_increment * i
                 for i in range(left_max - 1, -1, -1)]
            ),
            axis=0,
        )
        sorted_indices = np.argsort(
            ranges_legal * np.abs(np.cos(ranges_legal_ang))
        )
        sorted_indices = np.where(
            sorted_indices >= right_max,
            sorted_indices + left_right_inc,
            sorted_indices,
        )
        min_distance = float('inf')
        if self.last_scan is not None:
            for i in sorted_indices:
                temp_min = ranges[i]
                side_distance = temp_min * np.abs(
                    np.sin(scan_data.angle_increment * i + ang_min)
                )
                if side_distance < 0.5:
                    window_index = np.clip(
                        [i - self.window_size, i + self.window_size + 1],
                        0,
                        len(self.last_scan),
                    )
                    window = self.last_scan[window_index[0]:window_index[1]]
                    with np.errstate(invalid='ignore'):
                        if np.any(
                            np.abs(window - temp_min) <= self.distance_tolerance
                        ):
                            min_distance = temp_min * np.abs(
                                np.cos(scan_data.angle_increment * i + ang_min)
                            )
                            break
        self.last_scan = ranges
        if state == 0:
            threshold = 3 * speed_levels[0]
        elif state == 10 or state == 11:
            threshold = 0.2
        else:
            threshold = 0.4
        if min_distance < threshold:
            if self.distance_avoidance < 2:
                self.distance_avoidance += 1
            else:
                self.stop_hold = True
                if min_distance < threshold * 0.7:
                    self.stopped = True
                    self.last_stop_time = time.time()
        else:
            self.distance_avoidance = 0
            self.stopped = False
        return {
            'min_distance': min_distance,
            'stopped': self.stopped,
            'stop_hold': self.stop_hold,
            'last_stop_time': self.last_stop_time
        }
