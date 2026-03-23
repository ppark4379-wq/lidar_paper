#!/usr/bin/env python3
import rospy
import cv2
import torch
import numpy as np
import time # 정밀 시간 측정용
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import os
import sys

# YOLOPv2 소스코드 경로 설정
current_path = os.path.dirname(os.path.abspath(__file__))
yolop_path = os.path.join(current_path, "YOLOPv2")
sys.path.insert(0, yolop_path)
from utils.utils import letterbox

class MaskInferenceTimer:
    def __init__(self):
        rospy.init_node('mask_inference_timer')
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 로드 (연구에 사용된 YOLOPv2 및 YOLOv8-seg) [cite: 48, 58]
        yolop_weight = os.path.join(current_path, "yolopv2.pt")
        self.yolop_model = torch.jit.load(yolop_weight, map_location=self.device)
        self.yolop_model.eval()
        
        # 실시간성을 위해 Nano 모델 사용 [cite: 49]
        self.yolo_seg_model = YOLO("yolo26n-seg.pt") 

        # 결과 발행 및 데이터 구독
        self.mask_pub = rospy.Publisher("/yolop_fusion_masks", Image, queue_size=1)
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback, queue_size=1, buff_size=2**24)
        
        self.time_records = []
        rospy.loginfo("🚀 [Python] Pure Logic Inference Measurement Started")
        rospy.spin()

    def callback(self, msg):
        # 1. 이미지 전처리 (측정 범위에서 제외하고 싶다면 위치 조정 가능)
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None: return
        h, w = img.shape[:2]

        # --- [⏱️ 딥러닝 추론 시간 측정 시작] ---
        if torch.cuda.is_available():
            torch.cuda.synchronize() # GPU 작업 동기화
        start_time = time.perf_counter()

        # 2. YOLOPv2 도로 영역 세그멘테이션 (R 채널용) [cite: 26, 50]
        resized_img, ratio, pad = letterbox(img, new_shape=640, stride=32)
        img_tensor = torch.from_numpy(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().to(self.device) / 255.0
        
        with torch.no_grad():
            _, seg, _ = self.yolop_model(img_tensor.unsqueeze(0))
        
        # 도로 마스크 생성 및 원본 크기 복원 [cite: 51, 52]
        da_mask = torch.argmax(seg, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        m_c = da_mask[int(pad[1]):da_mask.shape[0]-int(pad[1]), int(pad[0]):da_mask.shape[1]-int(pad[0])]
        road_mask = (cv2.resize(m_c, (w, h), interpolation=cv2.INTER_NEAREST) == 1).astype(np.uint8) * 255

        # 3. YOLOv8 객체 세그멘테이션 (G 채널용) [cite: 26, 59]
        # 연구에 필요한 클래스(사람, 차량 등)만 추출 [cite: 58, 60]
        results = self.yolo_seg_model(img, classes=[0, 1, 2, 3, 5, 7], verbose=False)
        obs_mask = np.zeros((h, w), dtype=np.uint8)
        if results[0].masks is not None:
            for poly in results[0].masks.xy:
                cv2.fillPoly(obs_mask, [np.array(poly, np.int32)], 255)

        # 4. 최종 Semantic Fusion Mask 생성 [cite: 92, 93, 95]
        fusion_mask = np.zeros((h, w, 3), dtype=np.uint8)
        fusion_mask[:, :, 0] = road_mask # Blue 채널 (C++에서 [0]번 인덱스로 접근) [cite: 96]
        fusion_mask[:, :, 1] = obs_mask  # Green 채널 (C++에서 [1]번 인덱스로 접근)

        # --- [⏱️ 딥러닝 추론 시간 측정 종료] ---
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # 결과 계산 및 로그 출력
        duration_ms = (end_time - start_time) * 1000
        self.time_records.append(duration_ms)
        
        if len(self.time_records) % 30 == 0:
            avg_time = np.mean(self.time_records[-30:])
            rospy.loginfo(f"📊 Avg DL Inference: {avg_time:.2f} ms (Points extraction ready)")

        # 5. 마스크 발행 (C++ 노드로 전송) [cite: 55, 63]
        mask_msg = self.bridge.cv2_to_imgmsg(fusion_mask, "bgr8")
        mask_msg.header = msg.header # 원본 타임스탬프 유지
        self.mask_pub.publish(mask_msg)

if __name__ == '__main__':
    try:
        MaskInferenceTimer()
    except rospy.ROSInterruptException:
        pass