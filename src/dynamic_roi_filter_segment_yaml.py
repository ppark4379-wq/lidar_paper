#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamic ROI Filter (FINAL) - segments.yaml mode switching + paper-style x-index boundary
- LK: front only
- LC: front + back (back horizon automatically computed from rx_back)
- INT: ego-centered circular ROI (360deg)

Inputs:
  /global_path (nav_msgs/Path)
  /Ego_topic   (morai_msgs/EgoVehicleStatus)
  /lidar3D     (sensor_msgs/PointCloud2)   # param: ~lidar_topic

Outputs:
  /filtered_points (sensor_msgs/PointCloud2)
  /roi_boundary_marker (visualization_msgs/Marker)  # optional
  /roi_mode_marker     (visualization_msgs/Marker)  # optional
"""

import os
import math
import yaml
import rospy
import numpy as np

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Path
from morai_msgs.msg import EgoVehicleStatus

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class DynamicROIFilterSegmentYAMLFinal:
    def __init__(self):
        rospy.init_node("dynamic_roi_filter_segment_yaml_final")
        rospy.loginfo("[dynamic_roi_filter_final] start")

        # ----- topics -----
        self.lidar_topic = rospy.get_param("~lidar_topic", "/lidar3D")
        self.path_topic  = rospy.get_param("~path_topic",  "/global_path")
        self.ego_topic   = rospy.get_param("~ego_topic",   "/Ego_topic")

        # ----- segments.yaml -----
        self.segments_yaml = os.path.expanduser(
            rospy.get_param("~segments_yaml", "~/lidar_paper_ws/src/path/segments.yaml")
        )
        self.segments = self.load_segments(self.segments_yaml)

        # ----- base params (paper-ish) -----
        self.t_detect = float(rospy.get_param("~t_detect", 0.05))
        self.t_delay  = float(rospy.get_param("~t_delay", 0.30))
        self.tg_safe  = float(rospy.get_param("~tg_safe", 2.00))
        self.a_brake  = float(rospy.get_param("~a_brake", 1.50))
        self.c_s      = float(rospy.get_param("~c_s", 10.0))
        self.w_lane   = float(rospy.get_param("~w_lane", 3.5))
        self.i_x      = float(rospy.get_param("~i_x", 0.5))  # x-index resolution (m)

        # mode preview: "교차로 지나고 모드 바뀌는" 문제 해결용
        self.mode_preview_pts = int(rospy.get_param("~mode_preview_pts", 40))

        # path segment size (nearest부터 전방 몇 포인트 사용할지)
        self.forward_horizon_points = int(rospy.get_param("~forward_horizon_points", 350))

        # LC 후방 포인트 자동 계산용 (waypoint 평균 간격 가정값)
        self.ds_assume = float(rospy.get_param("~ds_assume", 0.3))  # m
        self.back_margin_pts = int(rospy.get_param("~back_margin_pts", 20))  # 여유 포인트
        self.min_back_pts = int(rospy.get_param("~min_back_pts", 5))
        self.max_back_pts = int(rospy.get_param("~max_back_pts", 800))

        # INT 원형 ROI 반경(차량 중심 기준)
        self.int_radius = float(rospy.get_param("~int_radius", 25.0))

        # ----- mode params (튜닝 포인트) -----
        # LK: 전방만 -> rx_back_mul=0.0 권장
        # LC: 전후방 -> rx_back_mul>=2.0 권장
        self.mode_params = rospy.get_param("~mode_params", {
            "LK":  {"ry_mul": 2.5, "rx_back_mul": 0.0, "rx_front_scale": 1.0},
            "LC":  {"ry_mul": 3.5, "rx_back_mul": 2.5, "rx_front_scale": 1.0},
            "INT": {"ry_mul": 5.0, "rx_back_mul": 1.0, "rx_front_scale": 1.0},  # INT는 int_radius 사용
        })

        # marker publish
        self.publish_markers = bool(rospy.get_param("~publish_markers", True))
        self.marker_frame = rospy.get_param("~marker_frame", "base_link")  # ego frame 기준

        # ----- runtime state -----
        self.has_ego = False
        self.has_path = False

        self.ego_v = 0.0
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_yaw = 0.0

        self.path_xy = None  # world path (N,2)

        self.boundary_array = None  # shape (L,2): [left, right]
        self.rx_front = None
        self.rx_back  = None

        self.mode = "LK"
        self.nearest_idx = 0
        self.mode_idx = 0

        # ----- ROS I/O -----
        self.sub_lidar = rospy.Subscriber(self.lidar_topic, PointCloud2, self.cb_lidar, queue_size=1)
        self.sub_path  = rospy.Subscriber(self.path_topic,  Path,       self.cb_path,  queue_size=1)
        self.sub_ego   = rospy.Subscriber(self.ego_topic,   EgoVehicleStatus, self.cb_ego, queue_size=1)

        self.pub_points = rospy.Publisher("/filtered_points", PointCloud2, queue_size=1)

        # debug markers
        self.pub_roi_marker = rospy.Publisher("/roi_boundary_marker", Marker, queue_size=1)
        self.pub_mode_marker = rospy.Publisher("/roi_mode_marker", Marker, queue_size=1)

        rospy.loginfo("[dynamic_roi_filter_final] segments_yaml=%s", self.segments_yaml)
        rospy.loginfo("[dynamic_roi_filter_final] mode_preview_pts=%d int_radius=%.1f", self.mode_preview_pts, self.int_radius)

    # ---------- util ----------
    def load_segments(self, yaml_path):
        if not os.path.exists(yaml_path):
            rospy.logwarn("[dynamic_roi_filter_final] segments.yaml not found: %s (fallback: LK all)", yaml_path)
            return [{"name": "LK", "start": 0, "end": 999999999}]
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}
        segs = data.get("segments", [])
        if not segs:
            segs = [{"name": "LK", "start": 0, "end": 999999999}]
        return sorted(segs, key=lambda s: int(s.get("start", 0)))

    def determine_mode(self, idx):
        for seg in self.segments:
            s = int(seg.get("start", 0))
            e = int(seg.get("end", 0))
            if s <= idx <= e:
                return str(seg.get("name", "LK"))
        return "LK"

    def compute_base_rx_front(self):
        v = self.ego_v
        rx = v * (self.t_detect + self.t_delay + self.tg_safe) + (v * v) / (2.0 * self.a_brake) + self.c_s
        return float(max(self.c_s, rx))

    def compute_nearest_idx(self):
        dx = self.path_xy[:, 0] - self.ego_x
        dy = self.path_xy[:, 1] - self.ego_y
        return int(np.argmin(dx * dx + dy * dy))

    # ---------- callbacks ----------
    def cb_ego(self, msg):
        self.ego_v = float(msg.velocity.x)
        self.ego_x = float(msg.position.x)
        self.ego_y = float(msg.position.y)
        self.ego_yaw = math.radians(float(msg.heading))  # MORAI heading: deg -> rad
        self.has_ego = True

        if self.has_path:
            self.rebuild_roi()

    def cb_path(self, msg):
        if len(msg.poses) < 2:
            return
        xy = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.path_xy = np.array(xy, dtype=np.float32)
        self.has_path = True

        if self.has_ego:
            self.rebuild_roi()

    # ---------- ROI core ----------
    def rebuild_roi(self):
        if not (self.has_ego and self.has_path):
            return
        if self.path_xy is None or self.path_xy.shape[0] < 2:
            return

        # 1) nearest idx
        self.nearest_idx = self.compute_nearest_idx()

        # 2) mode decision using preview idx (enter earlier!)
        self.mode_idx = min(self.nearest_idx + self.mode_preview_pts, self.path_xy.shape[0] - 1)
        self.mode = self.determine_mode(self.mode_idx)

        # 3) build ROI per mode
        if self.mode == "INT":
            # ---- 교차로: 차량 중심 원형 ROI ----
            R = self.int_radius
            self.rx_front = R
            self.rx_back = R
            self.boundary_array = self.build_circle_boundary(R)

        else:
            # ---- LK/LC: 경로 기반 리본 ROI ----
            mp = self.mode_params.get(self.mode, self.mode_params["LK"])
            ry = float(mp.get("ry_mul", 2.5)) * self.w_lane

            base_rx_front = self.compute_base_rx_front()
            rx_front = base_rx_front * float(mp.get("rx_front_scale", 1.0))
            rx_back  = self.c_s * float(mp.get("rx_back_mul", 1.0))

            # LK는 확실히 "전방만"
            if self.mode == "LK":
                rx_back = 0.0

            self.rx_front = float(rx_front)
            self.rx_back  = float(rx_back)

            # --- 핵심 수정: LC일 때 뒤쪽 path 포인트를 rx_back 기반으로 자동 확보 ---
            back_pts = self.min_back_pts
            if self.mode == "LC":
                # 대략 (rx_back / ds_assume) 만큼 포인트가 필요 + margin
                back_pts = int(self.rx_back / max(1e-6, self.ds_assume)) + self.back_margin_pts
                back_pts = max(self.min_back_pts, min(self.max_back_pts, back_pts))

            start = max(0, self.nearest_idx - back_pts)
            end   = min(self.path_xy.shape[0], self.nearest_idx + self.forward_horizon_points)
            seg = self.path_xy[start:end, :]

            # world -> ego (base_link) 변환
            cy = math.cos(self.ego_yaw)
            sy = math.sin(self.ego_yaw)
            dx = seg[:, 0] - self.ego_x
            dy = seg[:, 1] - self.ego_y
            lx = dx * cy + dy * sy
            ly = -dx * sy + dy * cy
            local = np.stack([lx, ly], axis=1).astype(np.float32)

            # x 범위로 자르기
            mask = (local[:, 0] >= -self.rx_back) & (local[:, 0] <= self.rx_front)
            local = local[mask]
            if local.shape[0] < 2:
                self.boundary_array = None
                return

            # x 정렬 (interp 안정화)
            order = np.argsort(local[:, 0])
            local = local[order]

            x = local[:, 0]
            y = local[:, 1]

            # 중복 x 제거
            ux, uidx = np.unique(x, return_index=True)
            x = x[uidx]
            y = y[uidx]
            if len(x) < 2:
                self.boundary_array = None
                return

            self.boundary_array = self.build_path_boundary(
                path_x=x, path_y=y,
                rx_front=self.rx_front, rx_back=self.rx_back, ry=ry,
                mode=self.mode
            )

        rospy.loginfo_throttle(
            1.0,
            "[dynamic_roi_filter_final] mode=%s nearest_idx=%d mode_idx=%d rx_front=%.1f rx_back=%.1f",
            self.mode, self.nearest_idx, self.mode_idx,
            float(self.rx_front if self.rx_front is not None else -1),
            float(self.rx_back  if self.rx_back  is not None else -1),
        )

        if self.publish_markers:
            self.publish_roi_markers()

    def build_circle_boundary(self, R):
        """차량 중심(0,0) 기준 원형 ROI: x in [-R, +R]"""
        length = int(math.floor((2.0 * R) / self.i_x)) + 1
        b = np.full((length, 2), np.nan, dtype=np.float32)  # [left, right]
        for i in range(length):
            tx = (i * self.i_x) - R
            circle = math.sqrt(max(0.0, R * R - tx * tx))
            b[i, 0] = circle
            b[i, 1] = -circle
        return b

    def build_path_boundary(self, path_x, path_y, rx_front, rx_back, ry, mode):
        """경로 기반 리본 ROI: x-index boundary"""
        length = int(math.floor((rx_front + rx_back) / self.i_x)) + 1
        b = np.full((length, 2), np.nan, dtype=np.float32)  # [left, right]

        x_min = float(path_x[0])
        x_max = float(path_x[-1])

        for i in range(length):
            tx = (i * self.i_x) - rx_back

            # LK는 전방만 확실히
            if mode == "LK" and tx < 0.0:
                continue

            # 보간 범위 밖은 비활성
            if tx < x_min or tx > x_max:
                continue

            ty = float(np.interp(tx, path_x, path_y))
            y_left  = ty + ry
            y_right = ty - ry

            # 원형 클램프(선택): 끝단 부드럽게
            r = rx_front if tx >= 0.0 else rx_back
            if r <= 1e-6:
                continue
            circle = math.sqrt(max(0.0, r * r - tx * tx))

            # clip
            y_left  = max(-circle, min(circle, y_left))
            y_right = max(-circle, min(circle, y_right))

            left = max(y_left, y_right)
            right = min(y_left, y_right)
            b[i, 0] = left
            b[i, 1] = right

        if np.sum(~np.isnan(b[:, 0])) < 10:
            return None
        return b

    # ---------- LiDAR filter ----------
    def cb_lidar(self, msg):
        if self.boundary_array is None or self.rx_back is None:
            return

        points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)

        b = self.boundary_array
        rx_back = float(self.rx_back)
        i_x = float(self.i_x)

        filtered = []
        for x, y, z in points:
            idx = int(math.floor((x + rx_back) / i_x))
            if idx < 0 or idx >= b.shape[0]:
                continue
            left = b[idx, 0]
            right = b[idx, 1]
            if np.isnan(left) or np.isnan(right):
                continue
            if right <= y <= left:
                filtered.append((x, y, z))

        out = pc2.create_cloud_xyz32(msg.header, filtered)
        self.pub_points.publish(out)

    # ---------- RViz markers ----------
    def publish_roi_markers(self):
        if self.boundary_array is None or self.rx_back is None:
            return

        b = self.boundary_array
        rx_back = float(self.rx_back)

        left_pts = []
        right_pts = []

        for i in range(b.shape[0]):
            left = b[i, 0]
            right = b[i, 1]
            if np.isnan(left) or np.isnan(right):
                continue
            x = (i * self.i_x) - rx_back
            left_pts.append(Point(x=x, y=float(left), z=0.0))
            right_pts.append(Point(x=x, y=float(right), z=0.0))

        m = Marker()
        m.header.frame_id = self.marker_frame
        m.header.stamp = rospy.Time.now()
        m.ns = "roi"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.08

        # mode별 색
        if self.mode == "LK":
            m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 1.0, 0.2, 1.0
        elif self.mode == "LC":
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.8, 0.2, 1.0
        else:
            m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.6, 1.0, 1.0

        pts = []
        pts.extend(left_pts)
        pts.extend(reversed(right_pts))
        if len(left_pts) > 0:
            pts.append(left_pts[0])
        m.points = pts
        self.pub_roi_marker.publish(m)

        t = Marker()
        t.header.frame_id = self.marker_frame
        t.header.stamp = rospy.Time.now()
        t.ns = "roi_mode"
        t.id = 1
        t.type = Marker.TEXT_VIEW_FACING
        t.action = Marker.ADD
        t.pose.position.x = 0.0
        t.pose.position.y = 0.0
        t.pose.position.z = 2.0
        t.scale.z = 0.8
        t.color.r, t.color.g, t.color.b, t.color.a = 1.0, 1.0, 1.0, 1.0
        t.text = f"MODE={self.mode}  nearest={self.nearest_idx}  mode_idx={self.mode_idx}"
        self.pub_mode_marker.publish(t)


if __name__ == "__main__":
    try:
        node = DynamicROIFilterSegmentYAMLFinal()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass