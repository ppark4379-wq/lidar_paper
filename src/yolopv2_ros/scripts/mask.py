#!/usr/bin/env python3
import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import os
import sys

# YOLOPv2 경로 설정
#yolop_path = ""   #yolopv2.pt있는 폴더
#sys.path.insert(0, yolop_path)

current_path = os.path.dirname(os.path.abspath(__file__))

# 1. YOLOPv2 소스코드 경로 설정 (scripts/YOLOPv2)
yolop_path = os.path.join(current_path, "YOLOPv2")
sys.path.insert(0, yolop_path)
from utils.utils import letterbox

class MaskPublisher:
    def __init__(self):
        rospy.init_node('mask_publisher')
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 절대 경로 (확인된 경로 사용)
        #yolop_weight = ""  #yolopv2.pt있는 폴더
        yolop_weight = os.path.join(current_path, "yolopv2.pt")
        self.yolop_model = torch.jit.load(yolop_weight, map_location=self.device)
        self.yolop_model.eval()
        self.yolo_seg_model = YOLO("yolo26n-seg.pt") # 0.088s 딜레이 달성을 위해 Nano 사용

        self.mask_pub = rospy.Publisher("/yolop_fusion_masks", Image, queue_size=1)
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback, queue_size=1, buff_size=2**24)
        
        rospy.loginfo("✅ [Python] Mask Publisher Node Started")
        rospy.spin()

    def callback(self, msg):
        # 1. 이미지 디코딩
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None: return
        h, w = img.shape[:2]

        # 2. YOLOPv2 도로 영역 (R 채널)
        resized_img, ratio, pad = letterbox(img, new_shape=640, stride=32)
        img_tensor = torch.from_numpy(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().to(self.device) / 255.0
        with torch.no_grad():
            _, seg, _ = self.yolop_model(img_tensor.unsqueeze(0))
        
        da_mask = torch.argmax(seg, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        m_c = da_mask[int(pad[1]):da_mask.shape[0]-int(pad[1]), int(pad[0]):da_mask.shape[1]-int(pad[0])]
        road_mask = (cv2.resize(m_c, (w, h), interpolation=cv2.INTER_NEAREST) == 1).astype(np.uint8) * 255

        # 3. YOLOv8 장애물 영역 (G 채널)
        results = self.yolo_seg_model(img, classes=[0, 1, 2, 3, 5, 7], verbose=False) # 0:사람, 1:자전거, 2:승용차, 3:오토바이, 5:버스, 7:트럭
        obs_mask = np.zeros((h, w), dtype=np.uint8)
        if results[0].masks is not None:
            for poly in results[0].masks.xy:
                cv2.fillPoly(obs_mask, [np.array(poly, np.int32)], 255)

        # 4. Visualize
        visual_img = img.copy()
        
        road_color = [0, 255, 0]     
        obstacle_color = [0, 0, 255]  

        color_mask = np.zeros_like(img)
        color_mask[road_mask == 255] = road_color
        color_mask[obs_mask == 255] = obstacle_color
        
        alpha = 1.0 # 원본의 선명도
        beta = 0.4  # 색상 마스크의 투명도(0.0~1.0)
        overlay_result = cv2.addWeighted(visual_img, alpha, color_mask, beta, 0)
        
        cv2.imshow("Visualizer", overlay_result)
        cv2.waitKey(1)

        # 5. 마스크 합성 및 발행
        fusion_mask = np.zeros((h, w, 3), dtype=np.uint8)
        fusion_mask[:, :, 0] = road_mask # Blue 채널이지만 C++에서 [0]으로 읽으므로 OK
        fusion_mask[:, :, 1] = obs_mask
        
        # 함수명 수정: cv2_to_imgmsg
        mask_msg = self.bridge.cv2_to_imgmsg(fusion_mask, "bgr8")
        mask_msg.header = msg.header # 동기화를 위해 원본 타임스탬프 유지
        self.mask_pub.publish(mask_msg)

if __name__ == '__main__':
    MaskPublisher()