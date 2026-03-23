#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from morai_msgs.msg import EgoVehicleStatus
import os

class PathMaker:
    def __init__(self):
        rospy.init_node('path_maker', anonymous=True)
        
        # 1. 저장할 기본 디렉토리 설정
        base_dir = os.path.expanduser('~/catkin_ws/src/lidar_paper')
        
        # 디렉토리가 없으면 생성 (에러 방지)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # 2. 중복되지 않는 파일 이름 찾기 (path1, path2, ...)
        count = 1
        while True:
            file_name = "path{0}.txt".format(count)
            full_path = os.path.join(base_dir, file_name)
            
            if not os.path.exists(full_path):
                self.file_path = full_path
                break
            count += 1
        
        # 3. 파일 열기
        self.f = open(self.file_path, 'w')
        
        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.status_callback)
        
        self.prev_x = 0
        self.prev_y = 0
        rospy.loginfo("--- 경로 기록 시작! [%s] 에 저장됩니다. ---", self.file_path)

    def status_callback(self, msg):
        curr_x = msg.position.x
        curr_y = msg.position.y
        curr_z = msg.position.z
        
        # 이전 좌표와 거리를 계산해서 0.5m 이상 이동했을 때만 기록
        dist = ((curr_x - self.prev_x)**2 + (curr_y - self.prev_y)**2)**0.5
        
        if dist > 0.5:
            # \t (탭)으로 구분하여 저장
            data = "{0}\t{1}\t{2}\n".format(curr_x, curr_y, curr_z)
            self.f.write(data)
            self.prev_x, self.prev_y = curr_x, curr_y
            print("기록 중... ({0}) x: {1:.2f}, y: {2:.2f}".format(os.path.basename(self.file_path), curr_x, curr_y))

    def shutdown(self):
        if hasattr(self, 'f'):
            self.f.close()
            rospy.loginfo("--- 경로 기록 완료! 파일 저장됨: %s ---", self.file_path)

if __name__ == '__main__':
    try:
        maker = PathMaker()
        rospy.on_shutdown(maker.shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass