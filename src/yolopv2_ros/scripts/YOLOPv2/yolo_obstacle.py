#!/usr/bin/env python3
import os
import sys
import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import CompressedImage, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
import tf.transformations as tf_trans
from std_msgs.msg import Header
current_dir = os.path.dirname(os.path.abspath(__file__))
yolop_path = "/home/autonav/lidar_paper_ws/yolopv2_code"
sys.path.insert(0, yolop_path)
from utils.utils import letterbox
# [추가] YOLOv8 라이브러리 임포트
from ultralytics import YOLO

# yolopv2_code 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "../yolopv2_code"))


class YOLOPv2LidarFusionNode:
    def __init__(self):
        rospy.loginfo("🚀 투 트랙 시맨틱 퓨전 노드 시작 (YOLOPv2 + YOLOv8n-seg)")
        
        self.K = np.array([
            [320.0,   0.0, 320.0],
            [  0.0, 320.0, 240.0],
            [  0.0,   0.0,   1.0]
        ])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ===== 1. 모델 로드 (YOLOPv2 & YOLOv8n-seg) =====
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # YOLOPv2 (주행 가능 영역용)
        self.yolop_path = rospy.get_param("~model_path", "../yolopv2_code/yolopv2.pt")
        self.yolop_model = torch.jit.load(self.yolop_path, map_location=self.device)
        self.yolop_model.eval()

        # YOLOv8-seg (장애물 마스크용 - 가장 가벼운 nano 모델 사용)
        rospy.loginfo("YOLOv8s-seg 모델 로딩 중... (최초 실행 시 자동 다운로드 됩니다)")
        self.yolo_seg_model = YOLO("yolov8s-seg.pt") 

        # ===== 2. Publisher 2개 분리 (도로 / 장애물) =====
        self.road_pub = rospy.Publisher("/lidar_road", PointCloud2, queue_size=1)
        self.obs_pub = rospy.Publisher("/lidar_obstacle", PointCloud2, queue_size=1)

        self.st_BGRImg = None
        self.latest_lidar_msg = None

        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback_image, queue_size=1)
        rospy.Subscriber("/lidar3D", PointCloud2, self.callback_lidar, queue_size=1)

        self.rate = rospy.Rate(10)
        self.img_size = 640
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
            return None, None

    def preprocess_yolop(self, img):
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
            img_h, img_w = image.shape[:2]

            # ==========================================
            # 1. YOLOPv2 추론 (초록색 바닥 마스크)
            # ==========================================
            input_tensor, ratio, pad = self.preprocess_yolop(image)
            with torch.no_grad():
                [pred, anchor_grid], seg, ll = self.yolop_model(input_tensor)
            
            da_mask = torch.argmax(seg, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
            drivable_mask = (da_mask == 1).astype(np.uint8) * 255
            road_mask_2d = self.restore_mask(drivable_mask, ratio, pad, image.shape)

            # ==========================================
            # 2. YOLOv8n-seg 추론 (장애물 마스크 생성)
            # ==========================================
            seg_results = self.yolo_seg_model(image, verbose=False)
            obs_mask_2d = np.zeros((img_h, img_w), dtype=np.uint8)
            
            # YOLO가 객체의 다각형(Polygon) 좌표를 반환하면, 이를 하얀색(255)으로 칠해 장애물 마스크 완성
            if seg_results[0].masks is not None:
                for poly in seg_results[0].masks.xy:
                    pts = np.array(poly, np.int32)
                    cv2.fillPoly(obs_mask_2d, [pts], 255)

            # ==========================================
            # 3. 라이다 3D -> 2D 투영 및 두 갈래 필터링
            # ==========================================
            # TF 프레임 맞춰둠
            R, t = self.get_rt_matrix("Camera-4", "Lidar3D-1")

            if R is not None:
                points_list = list(pc2.read_points(self.latest_lidar_msg, field_names=("x", "y", "z"), skip_nans=True))
                if len(points_list) == 0:
                    continue
                
                points_np = np.array(points_list)
                
                # Z축 제한 약간 완화 (도로 바닥부터 차 지붕까지 여유롭게)
                valid_3d_mask = (points_np[:, 2] > -2.0) & (points_np[:, 2] < 10.0)
                points_np = points_np[valid_3d_mask]
                
                pt_cam_raw = np.dot(R, points_np.T) + t
                x_c, y_c, z_c = -pt_cam_raw[1, :], -pt_cam_raw[2, :], pt_cam_raw[0, :]

                # 전방 점들만 (카메라 앞)
                front_mask = (z_c > 0.1) & (z_c < 50.0)
                x_c, y_c, z_c = x_c[front_mask], y_c[front_mask], z_c[front_mask]
                points_3d = points_np[front_mask]

                # 픽셀 좌표 변환
                u = ((self.K[0,0] * x_c / z_c) + self.K[0,2]).astype(np.int32)
                v = ((self.K[1,1] * y_c / z_c) + self.K[1,2]).astype(np.int32)

                in_img_mask = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
                u_valid, v_valid = u[in_img_mask], v[in_img_mask]
                points_3d_valid = points_3d[in_img_mask]

                # ⭐ 핵심: 마스크 위상에 따라 라이다 점을 두 그룹으로 쪼개기 ⭐
                is_obs = obs_mask_2d[v_valid, u_valid] > 0
                is_road = (road_mask_2d[v_valid, u_valid] > 0) & (~is_obs) # 장애물과 겹치는 바닥은 장애물 우선!

                obstacle_points = points_3d_valid[is_obs]
                road_points = points_3d_valid[is_road]

                result_img = image.copy()

                # 시각화 (초록색 = 길, 빨간색 = 장애물)
                road_idx = road_mask_2d > 0
                obs_idx = obs_mask_2d > 0
                result_img[road_idx] = (result_img[road_idx] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
                result_img[obs_idx] = (result_img[obs_idx] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
                for px, py in zip(u_valid[is_road], v_valid[is_road]):
                    cv2.circle(result_img, (px, py), 2, (0, 255, 0), -1) # 길은 초록 점
                for px, py in zip(u_valid[is_obs], v_valid[is_obs]):
                    cv2.circle(result_img, (px, py), 2, (0, 0, 255), -1) # 장애물은 빨간 점

                # ROS 퍼블리시
                header = Header()
                header.stamp = rospy.Time.now()
                header.frame_id = self.latest_lidar_msg.header.frame_id
                
                if len(road_points) > 0:
                    road_cloud = pc2.create_cloud_xyz32(header, road_points.tolist())
                    self.road_pub.publish(road_cloud)
                    
                if len(obstacle_points) > 0:
                    obs_cloud = pc2.create_cloud_xyz32(header, obstacle_points.tolist())
                    self.obs_pub.publish(obs_cloud)

                cv2.imshow("Two-Track Semantic Fusion", result_img)
                cv2.waitKey(1)

            self.rate.sleep()

if __name__ == "__main__":
    rospy.init_node('two_track_semantic_fusion', anonymous=True)
    try:
        YOLOPv2LidarFusionNode()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()