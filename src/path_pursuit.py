#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
import numpy as np

from nav_msgs.msg import Path
from morai_msgs.msg import EgoVehicleStatus
from morai_msgs.msg import CtrlCmd

class PurePursuitMORAI:
    def __init__(self):
        rospy.init_node("pure_pursuit_morai")

        # ====== params ======
        self.wheelbase = rospy.get_param("~wheelbase", 2.7)   # 차량 축거(대략값)
        self.min_ld = rospy.get_param("~min_ld", 5.0)         # 최소 전방주시거리
        self.k_ld = rospy.get_param("~k_ld", 0.7)             # Ld = min_ld + k_ld * v
        self.max_ld = rospy.get_param("~max_ld", 20.0)        # 최대 전방주시거리

        self.target_speed = rospy.get_param("~target_speed", 4.0)  # m/s (원하는 속도)
        self.kp_speed = rospy.get_param("~kp_speed", 0.3)

        # ====== state ======
        self.path = None              # (N,2) map 좌표
        self.ego = None               # EgoVehicleStatus
        self.last_nearest_idx = 0

        # ====== sub/pub ======
        self.sub_path = rospy.Subscriber("/global_path", Path, self.cb_path, queue_size=1)
        self.sub_ego  = rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.cb_ego, queue_size=1)

        self.pub_cmd = rospy.Publisher("/ctrl_cmd_0", CtrlCmd, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(0.02), self.run)  # 50Hz

        rospy.loginfo("[PP] pure pursuit controller started.")

    def cb_path(self, msg):
        if len(msg.poses) < 2:
            self.path = None
            return
        pts = []
        for ps in msg.poses:
            pts.append([ps.pose.position.x, ps.pose.position.y])
        self.path = np.array(pts, dtype=np.float32)
        self.last_nearest_idx = 0
        rospy.loginfo_throttle(1.0, "[PP] path received: %d points", len(pts))

    def cb_ego(self, msg):
        self.ego = msg

    def get_speed(self):
        # vx,vy가 있을 수 있으니 norm이 안전
        vx = self.ego.velocity.x
        vy = self.ego.velocity.y
        return math.sqrt(vx*vx + vy*vy)

    def compute_ld(self, v):
        ld = self.min_ld + self.k_ld * v
        return max(self.min_ld, min(self.max_ld, ld))

    def find_nearest_index(self, ex, ey):
        # 가까운 지점을 매번 전체검색하면 느릴 수 있어, last idx 근처부터 찾는 방식
        if self.path is None:
            return None

        N = len(self.path)
        start = max(0, self.last_nearest_idx - 50)
        end   = min(N, self.last_nearest_idx + 200)

        seg = self.path[start:end]
        dx = seg[:,0] - ex
        dy = seg[:,1] - ey
        d2 = dx*dx + dy*dy
        local_min = int(np.argmin(d2))
        idx = start + local_min
        self.last_nearest_idx = idx
        return idx

    def find_target_point(self, nearest_idx, ex, ey, ld):
        # nearest_idx부터 앞으로 누적거리로 ld 이상 되는 점 찾기
        N = len(self.path)
        if nearest_idx is None:
            return None

        accum = 0.0
        prev = self.path[nearest_idx]
        for i in range(nearest_idx + 1, N):
            cur = self.path[i]
            accum += float(np.linalg.norm(cur - prev))
            if accum >= ld:
                return cur, i
            prev = cur

        # 끝까지 가도 없으면 마지막 점
        return self.path[-1], N-1

    def map_to_ego(self, tx, ty, ex, ey, yaw):
        # map -> ego(base_link) 변환
        dx = tx - ex
        dy = ty - ey
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        lx =  cy*dx + sy*dy
        ly = -sy*dx + cy*dy
        return lx, ly

    def run(self, _evt):
        if self.path is None or self.ego is None:
            return

        ex = self.ego.position.x
        ey = self.ego.position.y
        yaw = math.radians(self.ego.heading)  # MORAI heading deg -> rad
        v = self.get_speed()

        ld = self.compute_ld(v)

        nearest_idx = self.find_nearest_index(ex, ey)
        target, tidx = self.find_target_point(nearest_idx, ex, ey, ld)
        tx, ty = float(target[0]), float(target[1])

        # target point in ego frame
        lx, ly = self.map_to_ego(tx, ty, ex, ey, yaw)

        # 전방 점이 아니라면(가끔 뒤에 있는 점 잡힘) nearest를 앞으로 조금 밀어줌
        if lx < 0.5:
            nearest_idx = min(nearest_idx + 5, len(self.path)-1)
            target, tidx = self.find_target_point(nearest_idx, ex, ey, ld)
            tx, ty = float(target[0]), float(target[1])
            lx, ly = self.map_to_ego(tx, ty, ex, ey, yaw)

        # alpha / steering
        alpha = math.atan2(ly, lx)
        # pure pursuit curvature
        kappa = 2.0 * math.sin(alpha) / max(ld, 1e-3)
        delta = math.atan(self.wheelbase * kappa)

        # ===== speed control (아주 간단한 P 제어) =====
        speed_err = self.target_speed - v
        accel_cmd = self.kp_speed * speed_err  # +면 가속, -면 감속

        # MORAI CtrlCmd 세팅 (프로젝트마다 field 다를 수 있음)
        cmd = CtrlCmd()
        cmd.longlCmdType = 1  # 2: accel/brake 모드(환경 따라 다를 수 있음)
        cmd.steering = float(delta)

        if accel_cmd >= 0.0:
            cmd.accel = float(min(accel_cmd, 1.0))
            cmd.brake = 0.0
        else:
            cmd.accel = 0.0
            cmd.brake = float(min(-accel_cmd, 1.0))

        self.pub_cmd.publish(cmd)

        rospy.loginfo_throttle(
            0.5,
            "[PP] v=%.2f ld=%.2f steer=%.3f target_idx=%d",
            v, ld, delta, tidx
        )

if __name__ == "__main__":
    try:
        PurePursuitMORAI()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass