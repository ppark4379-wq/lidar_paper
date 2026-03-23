#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import math
import time
import select
import termios
import tty

import rospy
from morai_msgs.msg import EgoVehicleStatus


def wrap_to_pi(a: float) -> float:
    """angle wrap to [-pi, pi]"""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class KeyReader:
    """
    Non-blocking key reader for terminal.
    - Works in a normal Linux terminal.
    - If you run in a non-TTY environment, key input may not work.
    """
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def get_key(self):
        """Return one character if pressed, else None."""
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            return sys.stdin.read(1)
        return None

    def close(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


class PathRecorderWithMode:
    def __init__(self):
        rospy.init_node("path_record_with_mode")

        # --- params ---
        self.ego_topic = rospy.get_param("~ego_topic", "/Ego_topic")
        self.save_dir = os.path.expanduser(rospy.get_param("~save_dir", "~/lidar_paper_ws/src/path/"))
        self.filename = rospy.get_param("~filename", "path_with_mode.csv")
        self.save_every_meter = float(rospy.get_param("~save_every_meter", 0.3))  # distance-based sampling
        self.yaw_wrap = bool(rospy.get_param("~yaw_wrap", True))

        os.makedirs(self.save_dir, exist_ok=True)
        self.filepath = os.path.join(self.save_dir, self.filename)

        # --- state ---
        self.has_ego = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.speed = 0.0

        self.last_saved_x = None
        self.last_saved_y = None
        self.idx = 0

        self.mode = "LK"  # default
        self.running = True

        # --- ROS ---
        self.sub = rospy.Subscriber(self.ego_topic, EgoVehicleStatus, self.ego_cb, queue_size=1)

        # --- open file ---
        self.f = open(self.filepath, "w", newline="")
        self.writer = csv.writer(self.f)
        self.writer.writerow(["idx", "time", "x", "y", "yaw", "mode", "speed"])  # header
        self.f.flush()

        rospy.loginfo("[path_record_with_mode] saving to: %s", self.filepath)
        rospy.loginfo("[path_record_with_mode] keys: 1(LK) 2(LC) 3(INT) 0(NONE) q(quit)")
        rospy.loginfo("[path_record_with_mode] save_every_meter: %.2f", self.save_every_meter)

        # --- key reader ---
        self.key_reader = None
        try:
            self.key_reader = KeyReader()
        except Exception as e:
            rospy.logwarn("[path_record_with_mode] key input init failed: %s", str(e))
            rospy.logwarn("[path_record_with_mode] run without key input (mode fixed).")

        # --- main loop timer ---
        self.timer = rospy.Timer(rospy.Duration(0.02), self.loop_cb)  # 50 Hz loop

    def ego_cb(self, msg: EgoVehicleStatus):
        self.x = float(msg.position.x)
        self.y = float(msg.position.y)
        # MORAI heading is degrees
        yaw = math.radians(float(msg.heading))
        self.yaw = wrap_to_pi(yaw) if self.yaw_wrap else yaw
        self.speed = float(msg.velocity.x)
        self.has_ego = True

    def update_mode_by_key(self, k: str):
        if k == "1":
            self.mode = "LK"
            rospy.loginfo("[mode] LK")
        elif k == "2":
            self.mode = "LC"
            rospy.loginfo("[mode] LC")
        elif k == "3":
            self.mode = "INT"
            rospy.loginfo("[mode] INT")
        elif k == "0":
            self.mode = "NONE"
            rospy.loginfo("[mode] NONE")
        elif k in ("q", "Q"):
            rospy.loginfo("[path_record_with_mode] quit requested.")
            self.running = False

    def should_save(self):
        if self.last_saved_x is None:
            return True
        dx = self.x - self.last_saved_x
        dy = self.y - self.last_saved_y
        dist = math.sqrt(dx * dx + dy * dy)
        return dist >= self.save_every_meter

    def save_row(self):
        t = rospy.Time.now().to_sec()
        self.writer.writerow([self.idx, f"{t:.6f}", f"{self.x:.6f}", f"{self.y:.6f}", f"{self.yaw:.6f}", self.mode, f"{self.speed:.6f}"])
        self.f.flush()

        self.last_saved_x = self.x
        self.last_saved_y = self.y
        self.idx += 1

    def loop_cb(self, _event):
        if not self.running:
            self.shutdown()
            return

        # key input
        if self.key_reader is not None:
            k = self.key_reader.get_key()
            if k is not None:
                self.update_mode_by_key(k)

        # save waypoint
        if not self.has_ego:
            return
        if self.should_save():
            self.save_row()

    def shutdown(self):
        # stop timer
        try:
            self.timer.shutdown()
        except Exception:
            pass

        # close key reader
        try:
            if self.key_reader is not None:
                self.key_reader.close()
        except Exception:
            pass

        # close file
        try:
            if self.f is not None:
                self.f.close()
        except Exception:
            pass

        rospy.loginfo("[path_record_with_mode] saved %d points -> %s", self.idx, self.filepath)
        rospy.signal_shutdown("finished")


if __name__ == "__main__":
    try:
        node = PathRecorderWithMode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass