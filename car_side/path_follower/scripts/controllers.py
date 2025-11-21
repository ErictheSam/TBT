#!/usr/bin/env python3

import numpy as np
import rospy
import time

class SimplePID:
    """A simple PID controller implementation with support for vector-valued signals.

    Attributes:
        kp (numpy.ndarray): Proportional gains.
        ki (numpy.ndarray): Integral gains.
        kd (numpy.ndarray): Derivative gains.
        set_point (numpy.ndarray): Target value(s) to track.
        last_error (numpy.ndarray): Previous error for derivative calculation.
        integrator (numpy.ndarray): Accumulated integral term.
        last_time (float or None): Timestamp of the last update call.
    """
    def __init__(self, target, p, i, d):
        kp_size = np.size(p)
        ki_size = np.size(i)
        kd_size = np.size(d)
        target_size = np.size(target)
        if (kp_size != ki_size or ki_size != kd_size or
            (target_size == 1 and kp_size != 1) or
            (target_size != 1 and kp_size != target_size and kp_size != 1)):
            raise TypeError('input parameters shape is not compatable')
        rospy.loginfo(f'PID initialised with P:{p}, I:{i}, D:{d}')
        self.kp = np.array(p)
        self.ki = np.array(i)
        self.kd = np.array(d)
        self.set_point = np.array(target)
        self.last_error = 0
        self.integrator = 0
        self.last_time = None

    def update(self, current_value):
        """Compute PID control output based on the current measured value.

        Args:
            current_value (array-like): The current measured value(s) to compare
                against the set-point. Must have the same shape as the target
                provided during initialization.

        Returns:
            numpy.ndarray: The PID control output, with the same shape as the
                input current_value.
        """
        current_value = np.array(current_value)
        if np.size(current_value) != np.size(self.set_point):
            raise TypeError('current_value and target do not have the same shape')
        if self.last_time is None:
            self.last_time = time.time()
            return np.zeros(np.size(current_value))
        error = current_value - self.set_point
        if -0.1 < error[0] < 0.1:
            error[0] = 0
        if -0.001 < error[1] < 0.001:
            error[1] = 0
        p = error
        current_time = time.time()
        delta_t = current_time - self.last_time
        self.integrator = self.integrator + (error * delta_t)
        i = self.integrator
        d = (error - self.last_error) / delta_t
        self.last_error = error
        self.last_time = current_time
        return self.kp * p + self.ki * i + self.kd * d
