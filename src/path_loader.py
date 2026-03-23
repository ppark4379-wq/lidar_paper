#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import os
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class PathLoader:
    def __init__(self):
        rospy.init_node('path_loader')
        
        # ROI 필터 코드가 구독하는 토픽명과 맞춥니다.
        self.path_pub = rospy.Publisher('/global_path', Path, queue_size=1)
        
        # 파일 경로 (path_data.txt가 있는 실제 경로로 수정하세요!)
        self.file_path = os.path.expanduser('~/catkin_ws/src/lidar_paper/path10.txt')
        
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'map' # 모라이 기본 좌표계

    def load_path(self):
        if not os.path.exists(self.file_path):
            rospy.logerr("파일을 찾을 수 없어요! 경로를 확인해주세요: " + self.file_path)
            return False

        with open(self.file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # 탭(\t)으로 구분된 x, y, z 좌표를 나눕니다. 
                tmp = line.split('\t')
                if len(tmp) < 3: continue
                
                read_pose = PoseStamped()
                read_pose.header.frame_id = 'map' # 각 포즈에도 frame_id 추가
                read_pose.header.stamp = rospy.Time.now() # 현재 시간 주입
                read_pose.pose.position.x = float(tmp[0])
                read_pose.pose.position.y = float(tmp[1])
                read_pose.pose.position.z = float(tmp[2])
                self.path_msg.poses.append(read_pose)
        
        rospy.loginfo("총 %d개의 웨이포인트를 성공적으로 불러왔습니다!", len(self.path_msg.poses))
        return True

    def run(self):
        if self.load_path():
            rate = rospy.Rate(1) # 1초에 한 번씩 경로 발행
            while not rospy.is_shutdown():
                self.path_msg.header.stamp = rospy.Time.now()
                self.path_pub.publish(self.path_msg)
                rate.sleep()

if __name__ == '__main__':
    try:
        loader = PathLoader()
        loader.run()
    except rospy.ROSInterruptException:
        pass