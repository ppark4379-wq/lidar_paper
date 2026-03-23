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

class DualSemanticFusionNode:
    def __init__(self):
        rospy.loginfo("도로 & 장애물 3D 퓨전 노드 시작")
        
        # 1. 카메라 행렬 (Intrinsic K)
        self.K = np.array([
            [320.0,   0.0, 320.0],
            [  0.0, 320.0, 240.0],
            [  0.0,   0.0,   1.0]
        ])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bridge = CvBridge()

        # 데이터 저장 변수
        self.st_BGRImg = None
        self.latest_lidar_msg = None
        self.latest_road_mask = None 
        self.latest_obs_mask = None

        # 2. Publisher 설정 (도로용, 장애물용 분리)
        self.road_pub = rospy.Publisher("/lidar_road", PointCloud2, queue_size=1)
        self.obs_pub = rospy.Publisher("/lidar_obstacle", PointCloud2, queue_size=1)

        # 3. Subscriber 설정
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback_image, queue_size=1)
        rospy.Subscriber("/lidar3D", PointCloud2, self.callback_lidar, queue_size=1)
        # 듀얼 마스크 구독
        rospy.Subscriber("/road_mask", Image, self.callback_road_mask, queue_size=1)
        rospy.Subscriber("/obstacle_mask", Image, self.callback_obs_mask, queue_size=1)

        self.rate = rospy.Rate(10) 
        self.main_loop()

    def callback_image(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.st_BGRImg = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def callback_lidar(self, msg):
        self.latest_lidar_msg = msg

    def callback_road_mask(self, msg):
        self.latest_road_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def callback_obs_mask(self, msg):
        self.latest_obs_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

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
            # 1. 모든 데이터 수신 확인
            if any(v is None for v in [self.st_BGRImg, self.latest_lidar_msg, self.latest_road_mask, self.latest_obs_mask]):
                self.rate.sleep()
                continue

            image = self.st_BGRImg.copy()
            road_mask = self.latest_road_mask 
            obs_mask = self.latest_obs_mask
            H, W = image.shape[:2]

            # 2. 좌표 변환 행렬 획득
            R, t = self.get_rt_matrix("Camera-3", "lidar_link")

            if R is not None:
                # [Step 1] 모든 라이다 점 추출
                pts = np.array(list(pc2.read_points(self.latest_lidar_msg, field_names=("x", "y", "z"), skip_nans=True)))
                if len(pts) == 0: continue

                # [Step 2] 3D -> 카메라 좌표계 변환
                pt_cam_raw = (R @ pts.T) + t
                x_c, y_c, z_c = -pt_cam_raw[1], -pt_cam_raw[2], pt_cam_raw[0]

                # [Step 3] ⭐ 경고 해결의 핵심: 전방 유효 거리 필터링 (z_c > 0.1) ⭐
                # 거리가 0이거나 음수인 점을 여기서 미리 제거하여 0으로 나누는 상황을 방지합니다.
                front_mask = z_c > 0.1
                
                x_front = x_c[front_mask]
                y_front = y_c[front_mask]
                z_front = z_c[front_mask]
                pts_front = pts[front_mask] # 원본 3D 점들도 같은 크기로 맞춰줍니다.

                # [Step 4] 안전해진 투영 계산 (이제 RuntimeWarning이 발생하지 않아요!)
                u = ((self.K[0,0] * x_front) / z_front).astype(np.int32) + int(self.K[0,2])
                v = ((self.K[1,1] * y_front) / z_front).astype(np.int32) + int(self.K[1,2])
                
                # [Step 5] 이미지 경계 내의 점들만 최종 추출
                in_img_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                
                u_valid, v_valid = u[in_img_mask], v[in_img_mask]
                pts_valid = pts_front[in_img_mask]

                # [Step 6] 세맨틱 데이터 매칭
                on_road_idx = road_mask[v_valid, u_valid] > 0
                road_points = pts_valid[on_road_idx]

                on_obs_idx = obs_mask[v_valid, u_valid] > 0
                obs_points = pts_valid[on_obs_idx]

                # [Step 7] 결과 발행
                header = Header(stamp=rospy.Time.now(), frame_id=self.latest_lidar_msg.header.frame_id)
                
                if len(road_points) > 0:
                    self.road_pub.publish(pc2.create_cloud_xyz32(header, road_points.tolist()))
                if len(obs_points) > 0:
                    self.obs_pub.publish(pc2.create_cloud_xyz32(header, obs_points.tolist()))

                # [Step 8] 시각화
                for ui, vi in zip(u_valid[on_road_idx], v_valid[on_road_idx]):
                    cv2.circle(image, (ui, vi), 2, (0, 255, 0), -1)
                for ui, vi in zip(u_valid[on_obs_idx], v_valid[on_obs_idx]):
                    cv2.circle(image, (ui, vi), 2, (0, 0, 255), -1)

                cv2.imshow("Dual Track Fusion (Green:Road, Red:Obs)", image)
                cv2.waitKey(1)

            self.rate.sleep()

if __name__ == "__main__":
    rospy.init_node('dual_semantic_fusion', anonymous=True)
    try:
        DualSemanticFusionNode()
    except rospy.ROSInterruptException:
        pass