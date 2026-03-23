#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Path
from morai_msgs.msg import EgoVehicleStatus

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class DynamicROIFilter:
    def __init__(self):
        rospy.init_node('dynamic_roi_filter')
        rospy.loginfo("--- MORAI ROI 노드 (안정화 + 파란 경계선) ---")

        # ===== 파라미터 =====
        self.t_detect = 0.05
        self.t_delay = 0.3
        self.tg_safe = 2.0
        self.a_brake = 1.5
        self.c_s = 10.0
        self.w_lane = 3.5
        self.i_x = 0.5

        # 전방 경로 몇 개만 사용할지 (너 코스 규모에 따라 200~800)
        self.front_wp_count = rospy.get_param("~front_wp_count", 400)

        # 보간 안정화용 x-bin (i_x보다 작게)
        self.x_bin_eps = rospy.get_param("~x_bin_eps", 0.2)

        # rx_front가 너무 작아지는거 방지
        self.min_rx_front = rospy.get_param("~min_rx_front", 20.0)
        self.max_rx_front = rospy.get_param("~max_rx_front", 80.0)

        # ===== ego state =====
        self.ego_initialized = False
        self.ego_v = 0.0
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_yaw = 0.0

        # ROI boundary array: [N,2] => [y_upper, y_lower]
        self.boundary_array = None

        # Subscribers
        self.point_sub = rospy.Subscriber("/lidar3D", PointCloud2, self.point_cb, queue_size=1)
        self.path_sub  = rospy.Subscriber("/global_path", Path, self.path_cb, queue_size=1)
        self.ego_sub   = rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.ego_cb, queue_size=1)

        # Publishers
        self.point_pub = rospy.Publisher("/filtered_points", PointCloud2, queue_size=1)
        self.roi_marker_pub = rospy.Publisher("/roi_markers", MarkerArray, queue_size=1)

    # ---------------- ego ----------------
    def ego_cb(self, msg):
        # 속도는 norm으로 쓰는게 안전(음수 vx 방지)
        self.ego_v = math.sqrt(msg.velocity.x**2 + msg.velocity.y**2)
        self.ego_x = msg.position.x
        self.ego_y = msg.position.y
        self.ego_yaw = math.radians(msg.heading)

        self.ego_initialized = True
        rospy.loginfo_throttle(1.0, "[ego] x=%.2f y=%.2f v=%.2f yaw(deg)=%.1f",
                               self.ego_x, self.ego_y, self.ego_v, msg.heading)

    # ---------------- path -> local ----------------
    def path_cb(self, msg):
        if not self.ego_initialized:
            return
        if len(msg.poses) < 5:
            rospy.logwarn_throttle(1.0, "[path] poses too small: %d", len(msg.poses))
            return

        ex, ey = self.ego_x, self.ego_y

        # 1) ego와 가장 가까운 waypoint 찾기 (map 기준)
        min_i, min_d2 = 0, 1e18
        for i, ps in enumerate(msg.poses):
            dx = ps.pose.position.x - ex
            dy = ps.pose.position.y - ey
            d2 = dx*dx + dy*dy
            if d2 < min_d2:
                min_d2 = d2
                min_i = i

        # 2) 가까운 지점부터 전방 N개만 사용
        N = self.front_wp_count
        seg = msg.poses[min_i : min(min_i + N, len(msg.poses))]

        # 3) map -> ego(local) 변환
        cy = math.cos(self.ego_yaw)
        sy = math.sin(self.ego_yaw)
        local_path = []

        for wp in seg:
            dx = wp.pose.position.x - ex
            dy = wp.pose.position.y - ey
            lx =  cy*dx + sy*dy
            ly = -sy*dx + cy*dy
            local_path.append([lx, ly])

        local_path = np.array(local_path, dtype=np.float32)

        # 4) x 정렬 + 중복 x를 bin으로 평균내서 1개로 (interp 안정화)
        local_path = self._cleanup_path_for_interp(local_path)
        if local_path.shape[0] < 5:
            rospy.logwarn_throttle(1.0, "[path] cleaned path too small: %d", local_path.shape[0])
            return

        self.update_boundary(local_path)
        rospy.loginfo_throttle(1.0, "[path] boundary updated from %d wps (nearest=%d)", len(seg), min_i)

    def _cleanup_path_for_interp(self, local_path):
        # x 기준 정렬
        order = np.argsort(local_path[:, 0])
        p = local_path[order]
        xs = p[:, 0]
        ys = p[:, 1]

        # x가 거의 같은 점들은 y를 평균내서 하나로
        x_new, y_new = [], []
        eps = float(self.x_bin_eps)

        i = 0
        while i < len(xs):
            x0 = xs[i]
            j = i
            y_sum = 0.0
            cnt = 0
            while j < len(xs) and abs(xs[j] - x0) < eps:
                y_sum += float(ys[j])
                cnt += 1
                j += 1
            x_new.append(float(x0))
            y_new.append(y_sum / max(1, cnt))
            i = j

        return np.stack([np.array(x_new, dtype=np.float32),
                         np.array(y_new, dtype=np.float32)], axis=1)

    # ---------------- boundary building ----------------
    def update_boundary(self, local_path):
        rx_front = self.ego_v * (self.t_detect + self.t_delay + self.tg_safe) + \
                   (self.ego_v ** 2 / (2 * self.a_brake)) + self.c_s
        rx_front = max(self.min_rx_front, min(self.max_rx_front, rx_front))
        rx_back = self.c_s
        ry = 2.5 * self.w_lane

        length = int((rx_front + rx_back) / self.i_x) + 1
        b_array = np.zeros((length, 2), dtype=np.float32)

        # interp 입력 준비
        xs = local_path[:, 0]
        ys = local_path[:, 1]
        x_min = float(xs[0])
        x_max = float(xs[-1])

        for i in range(length):
            target_x = (i * self.i_x) - rx_back

            # interp 범위를 벗어나면 끝값 사용(튀는거 방지)
            if target_x <= x_min:
                target_y = float(ys[0])
            elif target_x >= x_max:
                target_y = float(ys[-1])
            else:
                target_y = float(np.interp(target_x, xs, ys))

            y_left = target_y + ry
            y_right = target_y - ry

            # 반원 제한
            r_limit = rx_front if target_x >= 0.0 else rx_back
            circle_y_limit = math.sqrt(max(0.0, (r_limit * r_limit) - (target_x * target_x)))

            upper = min(y_left,  circle_y_limit)
            lower = max(y_right, -circle_y_limit)

            if upper < lower:
                upper, lower = lower, upper

            b_array[i, 0] = upper
            b_array[i, 1] = lower

        self.boundary_array = b_array

    # ---------------- markers ----------------
    def publish_roi_markers(self, frame_id, rx_back):
        if self.boundary_array is None:
            return

        arr = MarkerArray()

        def make_line(mid, width):
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = rospy.Time.now()
            m.ns = "roi"
            m.id = mid
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = width
            # 파란색
            m.color.r = 0.0
            m.color.g = 0.3
            m.color.b = 1.0
            m.color.a = 1.0
            return m

        left_line = make_line(0, 0.08)
        right_line = make_line(1, 0.08)

        N = len(self.boundary_array)
        for i in range(N):
            x = i * self.i_x - rx_back

            p1 = Point()
            p1.x = float(x)
            p1.y = float(self.boundary_array[i, 0])
            p1.z = 0.0
            left_line.points.append(p1)

            p2 = Point()
            p2.x = float(x)
            p2.y = float(self.boundary_array[i, 1])
            p2.z = 0.0
            right_line.points.append(p2)

        arr.markers.append(left_line)
        arr.markers.append(right_line)

        self.roi_marker_pub.publish(arr)

    # ---------------- point filtering ----------------
    def point_cb(self, msg):
        if self.boundary_array is None:
            rospy.logwarn_throttle(1.0, "[point] boundary_array is None")
            return

        points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        filtered_points = []
        rx_back = self.c_s

        for p in points:
            idx = int((p[0] + rx_back) / self.i_x)
            if 0 <= idx < len(self.boundary_array):
                if self.boundary_array[idx, 1] <= p[1] <= self.boundary_array[idx, 0]:
                    filtered_points.append(p)

        out_msg = pc2.create_cloud_xyz32(msg.header, filtered_points)
        self.point_pub.publish(out_msg)

        # 파란 경계선 퍼블리시 (라이다 프레임 그대로)
        self.publish_roi_markers(msg.header.frame_id, rx_back)


if __name__ == '__main__':
    try:
        DynamicROIFilter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass