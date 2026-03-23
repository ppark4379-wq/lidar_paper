#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Path
from morai_msgs.msg import EgoVehicleStatus 

class DynamicROIFilter:
    def __init__(self):
        rospy.init_node('dynamic_roi_filter')
        rospy.loginfo("--- 성준님의 모라이 ROI 노드 가동! ---")

        # [1] 논문 기반 파라미터 설정
        self.t_detect = 0.05      # 인식 시간 
        self.t_delay = 0.3        # 제동 지연 
        self.tg_safe = 2.0        # 안전 여유 시간 
        self.a_brake = 1.5        # 제동 가속도 
        self.c_s = 10.0           # 정차 여유 거리 
        self.w_lane = 3.5         # 차선 너비 
        self.i_x = 0.5            # x축 인덱스 간격 

        self.ego_v = 0.0
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_yaw = 0.0
        self.boundary_array = None 

        # Subscriber: 확인하신 토픽명으로 변경!
        self.point_sub = rospy.Subscriber("/lidar3D", PointCloud2, self.point_cb)
        self.path_sub = rospy.Subscriber("/global_path", Path, self.path_cb)
        self.ego_sub = rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.ego_cb)
        
        self.point_pub = rospy.Publisher("/filtered_points", PointCloud2, queue_size=1)

    def ego_cb(self, msg):
        # echo로 확인하신 데이터 구조를 매핑합니다.
        self.ego_v = msg.velocity.x
        self.ego_x = msg.position.x
        self.ego_y = msg.position.y
        # 모라이 heading은 degree 단위이므로 radian으로 변환합니다.
        self.ego_yaw = math.radians(msg.heading)

    def path_cb(self, msg):
        if self.ego_x == 0.0: return
        
        # [2] 주행 경로 기반 ROI 생성 (전처리)
        local_path = []
        for wp in msg.poses:
            dx = wp.pose.position.x - self.ego_x
            dy = wp.pose.position.y - self.ego_y
            
            # 좌표계 변환: World -> Ego
            lx = dx * math.cos(self.ego_yaw) + dy * math.sin(self.ego_yaw)
            ly = -dx * math.sin(self.ego_yaw) + dy * math.cos(self.ego_yaw)
            local_path.append([lx, ly])
        
        if len(local_path) > 0:
            self.update_boundary(np.array(local_path))

    def update_boundary(self, local_path):
        # [1] 상황별 인지 거리(R) 결정 
        rx_front = self.ego_v * (self.t_detect + self.t_delay + self.tg_safe) + \
                   (self.ego_v**2 / (2 * self.a_brake)) + self.c_s
        rx_back = self.c_s
        ry = 2.5 * self.w_lane

        # [3] 인덱스 기반 경계 배열 생성 
        length = int((rx_front + rx_back) / self.i_x)
        b_array = np.zeros((length, 2)) 

        for i in range(length):
            target_x = (i * self.i_x) - rx_back
            # 경로 보간
            target_y = np.interp(target_x, local_path[:, 0], local_path[:, 1])
            
            y_left = target_y + ry # 오프셋 형성 
            y_right = target_y - ry

            # 원형 제한 
            r_limit = rx_front if target_x >= 0 else rx_back
            circle_y_limit = math.sqrt(max(0, r_limit**2 - target_x**2))
            
            b_array[i, 0] = min(y_left, circle_y_limit)
            b_array[i, 1] = max(y_right, -circle_y_limit)

        self.boundary_array = b_array

    def point_cb(self, msg):
        if self.boundary_array is None: return
        
        # [4] 최종 라이다 점 선별 
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

if __name__ == '__main__':
    try:
        DynamicROIFilter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass