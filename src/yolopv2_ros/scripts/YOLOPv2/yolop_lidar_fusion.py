#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, PointCloud2, Image
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
import tf2_ros
import tf.transformations as tf_trans
from std_msgs.msg import Header
import time

class YOLOPv2LidarFusionNode:
    def __init__(self):
        rospy.loginfo("Numpy 퓨전 노드")
        
        # 행렬 설정 (K 행렬)
        self.K = np.array([
            [320.0,   0.0, 320.0],
            [  0.0, 320.0, 240.0],
            [  0.0,   0.0,   1.0]
        ])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bridge = CvBridge()

        self.st_BGRImg = None
        self.latest_lidar_msg = None
        self.latest_mask = None 

        self.lidar_pub = rospy.Publisher("/lidar_roi", PointCloud2, queue_size=1)

        # Subscriber
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback_image, queue_size=1)
        rospy.Subscriber("/lidar3D", PointCloud2, self.callback_lidar, queue_size=1)
        rospy.Subscriber("/yolop_mask", Image, self.callback_mask, queue_size=1)

        self.rate = rospy.Rate(20) # 20Hz?
        self.main_loop()

    def callback_image(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.st_BGRImg = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def callback_lidar(self, msg):
        self.latest_lidar_msg = msg

    def callback_mask(self, msg):
        self.latest_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def get_rt_matrix(self, target_frame, source_frame):
        try:
            trans = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0))
            t = np.array([[trans.transform.translation.x], [trans.transform.translation.y], [trans.transform.translation.z]])
            q = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
            R = tf_trans.quaternion_matrix(q)[:3, :3]
            return R, t
        except: return None, None

    def main_loop(self):
        while not rospy.is_shutdown():
            if self.st_BGRImg is None or self.latest_lidar_msg is None or self.latest_mask is None:
                self.rate.sleep()
                continue

            image = self.st_BGRImg.copy()
            drivable_resized = self.latest_mask 
            H, W = image.shape[:2]

            R, t = self.get_rt_matrix("Camera-3", "lidar_link")

            if R is not None:
                # 1. 모든 라이다 점을 한꺼번에 Numpy 배열로 추출 (N, 3)
                pts = np.array(list(pc2.read_points(self.latest_lidar_msg, field_names=("x", "y", "z"), skip_nans=True)))
                if len(pts) == 0: continue

                # 2. 행렬 연산으로 투영 (R * P.T + t)
                # --- [수정된 투영 로직 부분] ---

                # 1. 3D 좌표 변환
                pt_cam_raw = (R @ pts.T) + t
                x_c, y_c, z_c = -pt_cam_raw[1], -pt_cam_raw[2], pt_cam_raw[0]

                # 2. ⭐ 중요: 투영 계산 전에 '앞에 있는 점들만' 먼저 필터링합니다!
                # z_c > 0.1 조건을 먼저 적용해서 0으로 나누는 상황을 물리적으로 차단해요.
                front_mask = z_c > 0.1 

                x_front = x_c[front_mask]
                y_front = y_c[front_mask]
                z_front = z_c[front_mask]
                pts_front = pts[front_mask]

                # 3. 이제 안전하게 투영 계산! (Warning이 사라집니다 ✨)
                u = ((self.K[0,0] * x_front) / z_front).astype(np.int32) + int(self.K[0,2])
                v = ((self.K[1,1] * y_front) / z_front).astype(np.int32) + int(self.K[1,2])

                # 4. 이미지 경계 내 필터링 (기존 로직 유지)
                in_img_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)

                u_valid, v_valid = u[in_img_mask], v[in_img_mask]
                pts_valid = pts_front[in_img_mask] # pts_front에서 골라내야 함!

                # 6. 도로 영역(Semantic ROI) 필터링
                on_road_mask = drivable_resized[v_valid, u_valid] > 0
                roi_points = pts_valid[on_road_mask]

                # 7. 결과 발행 (/lidar_roi)
                if len(roi_points) > 0:
                    header = Header(stamp=rospy.Time.now(), frame_id=self.latest_lidar_msg.header.frame_id)
                    self.lidar_pub.publish(pc2.create_cloud_xyz32(header, roi_points.tolist()))

                # 8. 로그 출력 
                rospy.loginfo(f"Total: {len(pts)} | Front: {np.sum(front_mask)} | In-Img: {np.sum(in_img_mask)} | ROI: {len(roi_points)}")

                # 9. 시각화 (동일한 파란색 점)
                u_road, v_road = u_valid[on_road_mask], v_valid[on_road_mask]
                # 시각화 속도를 위해 점 찍는 루프는 최소화
                for i in range(len(u_road)):
                    cv2.circle(image, (u_road[i], v_road[i]), 2, (255, 0, 0), -1)

                cv2.imshow("Lidar-Camera Fusion Result", image)
                cv2.waitKey(1)

            self.rate.sleep()

if __name__ == "__main__":
    rospy.init_node('yolopv2_lidar_fusion', anonymous=True)
    YOLOPv2LidarFusionNode()
    
'''
#!/usr/bin/env python3
import os
import sys
import rospy
import cv2
import torch
import numpy as np
from pathlib import Path
from sensor_msgs.msg import CompressedImage, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
import tf.transformations as tf_trans
import gc
from std_msgs.msg import Header # 헤더 추가

# yolopv2_code 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "../yolopv2_code"))
from utils.utils import (
    split_for_trace_model,
    lane_line_mask,
    non_max_suppression,
    letterbox
)

class YOLOPv2LidarFusionNode:
    def __init__(self):
        # ===== 1. ROS 초기화 및 설정 =====
        rospy.loginfo("퓨전 노드 시작")
        
        # 행렬 설정 (K 행렬)
        self.K = np.array([
            [320.0,   0.0, 320.0],
            [  0.0, 320.0, 240.0],
            [  0.0,   0.0,   1.0]
        ])

        # TF 리스너
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ===== 2. 모델 로드 =====
        self.model_path = rospy.get_param("~model_path", "../yolopv2_code/yolopv2.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(self.model_path, map_location=self.device)
        self.model.eval()

        # ===== 3. Subscriber & Publisher =====
        self.bridge = CvBridge()
        self.st_BGRImg = None
        self.latest_lidar_msg = None

        # [수정] 필터링된 점들을 발행할 Publisher 추가
        self.lidar_pub = rospy.Publisher("/lidar_roi", PointCloud2, queue_size=1)

        # 카메라 이미지 구독
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback_image, queue_size=1)
        # 라이다 데이터 구독
        rospy.Subscriber("/lidar3D", PointCloud2, self.callback_lidar, queue_size=1)

        # ===== 4. 실행 설정 =====
        self.rate = rospy.Rate(10) # 10Hz
        self.img_size = 640
        self.conf_thres = 0.8
        self.iou_thres = 0.8
        
        self.main_loop()

    def callback_image(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.st_BGRImg = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logerr(f"이미지 수신 오류: {e}")

    def callback_lidar(self, msg):
        self.latest_lidar_msg = msg

    def get_rt_matrix(self, target_frame, source_frame):
        try:
            trans = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0))
            t = np.array([[trans.transform.translation.x], [trans.transform.translation.y], [trans.transform.translation.z]])
            q = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
            R = tf_trans.quaternion_matrix(q)[:3, :3]
            return R, t
        except Exception as e:
            rospy.logwarn(f"TF 대기 중... : {e}")
            return None, None

    def preprocess(self, img):
        resized_img, ratio, pad = letterbox(img, new_shape=self.img_size, stride=32)
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        return img_tensor.unsqueeze(0).to(self.device), ratio, pad

    def restore_mask(self, mask, ratio, pad, original_shape):
        h_ori, w_ori = original_shape[:2]
        pad_w, pad_h = int(pad[0]), int(pad[1])
        h_mask, w_mask = mask.shape[:2]
        mask_cropped = mask[pad_h : h_mask - pad_h, pad_w : w_mask - pad_w]
        return cv2.resize(mask_cropped, (w_ori, h_ori), interpolation=cv2.INTER_NEAREST)

    def main_loop(self):
        while not rospy.is_shutdown():
            if self.st_BGRImg is None or self.latest_lidar_msg is None:
                self.rate.sleep()
                continue

            image = self.st_BGRImg.copy()
            input_tensor, ratio, pad = self.preprocess(image)
            
            with torch.no_grad():
                [pred, anchor_grid], seg, ll = self.model(input_tensor)
            
            da_mask = torch.argmax(seg, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
            drivable_mask = (da_mask == 1).astype(np.uint8) * 255
            drivable_resized = self.restore_mask(drivable_mask, ratio, pad, image.shape)

            R, t = self.get_rt_matrix("Camera-3", "lidar_link")

            if R is not None:
                points = pc2.read_points(self.latest_lidar_msg, field_names=("x", "y", "z"), skip_nans=True)
                result_img = image.copy()
                
                # [수정] ROI 점들을 담을 리스트 초기화
                roi_points = []
                total_pts = 0
                front_pts = 0
                in_img_pts = 0
                mask_pts = 0

                for p in points:
                    total_pts += 1
                    pt_lidar = np.array([[p[0]], [p[1]], [p[2]]])
                    pt_cam_raw = np.dot(R, pt_lidar) + t
                    
                    x_c = -pt_cam_raw[1][0]
                    y_c = -pt_cam_raw[2][0]
                    z_c =  pt_cam_raw[0][0]

                    if z_c <= 0.1: 
                        continue 
                    front_pts += 1

                    u = int((self.K[0,0] * x_c / z_c) + self.K[0,2])
                    v = int((self.K[1,1] * y_c / z_c) + self.K[1,2])

                    if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                        in_img_pts += 1
                        if drivable_resized[v, u] > 0:
                            mask_pts += 1
                            cv2.circle(result_img, (u, v), 2, (255, 0, 0), -1)
                            # [수정] 도로 영역 안에 있는 라이다 점을 리스트에 추가
                            roi_points.append([p[0], p[1], p[2]])

                # [수정] 수집된 ROI 점들을 PointCloud2 메시지로 변환하여 발행
                if len(roi_points) > 0:
                    header = Header()
                    header.stamp = rospy.Time.now()
                    header.frame_id = self.latest_lidar_msg.header.frame_id # lidar_link 사용
                    
                    roi_cloud = pc2.create_cloud_xyz32(header, roi_points)
                    self.lidar_pub.publish(roi_cloud)

                rospy.loginfo(f"Total: {total_pts} | Front: {front_pts} | In-Image: {in_img_pts} | On-Road: {mask_pts}")
                cv2.imshow("Lidar-Camera Fusion Result", result_img)
                cv2.waitKey(1)

            self.rate.sleep()

if __name__ == "__main__":
    rospy.init_node('yolopv2_lidar_fusion', anonymous=True)
    try:
        YOLOPv2LidarFusionNode()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
'''