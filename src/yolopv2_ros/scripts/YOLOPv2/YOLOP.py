#!/usr/bin/env python3
import os
import sys
import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import CompressedImage, Image # Image 추가
from cv_bridge import CvBridge, CvBridgeError
import torch, gc

# yolopv2_code 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "../yolopv2_code"))
from utils.utils import (
    split_for_trace_model,
    lane_line_mask,
    non_max_suppression,
    letterbox
)

class YOLOPv2ROSNode:
    def __init__(self):
        # ===== 1. ROS 통신 설정 =====
        # 퓨전 노드에 전달할 마스크 Publisher
        self.mask_pub = rospy.Publisher("/yolop_mask", Image, queue_size=1)
        self.bridge = CvBridge()
        
        # 이미지 Subscriber
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback_Image, queue_size=1, buff_size=2**24)

        # ===== 2. 모델 로드 =====
        self.model_path = rospy.get_param("~model_path", "../yolopv2_code/yolopv2.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"모델 로딩 중: {self.model_path}")
        self.model = torch.jit.load(self.model_path, map_location=self.device)
        self.model.eval()

        # ===== 3. 파라미터 설정 =====
        self.img_size = 640
        self.conf_thres = 0.8
        self.iou_thres = 0.8
        self.st_BGRImg = None
        self.rate = rospy.Rate(20) 

        rospy.on_shutdown(self.cleanup)
        self.YolopDetection()

    def cleanup(self):
        print("[INFO] Shutting down, closing OpenCV windows.")
        torch.cuda.empty_cache()
        gc.collect()
        cv2.destroyAllWindows()
    
    def callback_Image(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                self.st_BGRImg = img_bgr
        except Exception as e:
            rospy.logerr(f"이미지 변환 오류: {e}")

    def preprocess(self, img):
        resized_img, ratio, pad = letterbox(img, new_shape=self.img_size, stride=32)
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        return img_tensor.unsqueeze(0).to(self.device), ratio, pad
    
    def restore_mask(self, mask, ratio, pad, original_shape):
        h_ori, w_ori = original_shape[:2]
        if mask.shape[:2] == (h_ori, w_ori):
            return mask
        pad_w, pad_h = int(pad[0]), int(pad[1])
        h_mask, w_mask = mask.shape[:2]
        mask_cropped = mask[pad_h : h_mask - pad_h, pad_w : w_mask - pad_w]
        return cv2.resize(mask_cropped, (w_ori, h_ori), interpolation=cv2.INTER_NEAREST)

    def YolopDetection(self):
        while self.st_BGRImg is None:
            rospy.sleep(0.05)

        try:
            while not rospy.is_shutdown():
                image = self.st_BGRImg.copy()
                input_tensor, ratio, pad = self.preprocess(image)

                with torch.no_grad():
                    [pred, anchor_grid], seg, ll = self.model(input_tensor)

                # 1. 모델 마스크 생성
                da_mask = torch.argmax(seg, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
                drivable_mask = (da_mask == 1).astype(np.uint8) * 255
                drivable_resized = self.restore_mask(drivable_mask, ratio, pad, image.shape)

                # 2. 오버레이 및 합성 시각화 로직
                overlay = np.zeros_like(image)
                overlay[drivable_resized > 127] = [0, 255, 0]  # 초록색 도로
                result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0) # 합성 이미지

                # 3. 화면 표시
                #cv2.imshow("YOLOPv2", overlay)
                cv2.imshow("YOLOPv2 Result", result)

                # 4. 퓨전 노드 전송용 마스크 발행
                mask_msg = self.bridge.cv2_to_imgmsg(drivable_resized, encoding="mono8")
                self.mask_pub.publish(mask_msg)

                cv2.waitKey(1)
                self.rate.sleep()

        except Exception as e:
            rospy.logerr(f"[YOLOPv2 Node] 오류 발생: {e}")

if __name__ == "__main__":
    rospy.init_node('YOLOPv2', anonymous=True)
    YOLOPv2ROSNode()

'''
#!/usr/bin/env python3
import os
import sys
import rospy
import cv2
import torch
import numpy as np
from pathlib import Path
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge,CvBridgeError
import torch, gc

# yolopv2_code 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "../yolopv2_code"))
from utils.utils import (
    split_for_trace_model,
    driving_area_mask,
    lane_line_mask,
    non_max_suppression,
    show_seg_result,
    letterbox
)

class YOLOPv2ROSNode:
    def __init__(self):

        # ===== Subscriber =====
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback_Image, queue_size=1,buff_size=2**24)

        # ===== model =====
        self.model_path = rospy.get_param("~model_path", "../yolopv2_code/yolopv2.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"모델 로딩 중: {self.model_path}")
        self.model = torch.jit.load(self.model_path, map_location=self.device)
        self.model.eval()

        # ===== parameters =====
        self.last_process_time = rospy.Time.now()
        self.min_interval = rospy.Duration(0.05)  # 0.2초 (5 FPS)
        self.bridge = CvBridge()
        self.img_size = 640
        self.conf_thres = 0.8
        self.iou_thres = 0.8
        self.st_BGRImg = None

        # ===== process =====
        self.rate = rospy.Rate(20) 
        rospy.on_shutdown(self.cleanup)
        self.YolopDetection()

    def cleanup(self):
        print("[INFO] Shutting down, closing OpenCV windows.")
        torch.cuda.empty_cache()
        gc.collect()
        cv2.destroyAllWindows()
    
    def callback_Image(self,msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # 압축 해제 + BGR 변환
            if img_bgr is not None:
                self.st_BGRImg=img_bgr
        except CvBridgeError as e:
            print(e)

    def preprocess(self, img):
        resized_img, ratio, pad = letterbox(img, new_shape=self.img_size, stride=32)
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        return img_tensor.unsqueeze(0).to(self.device), ratio, pad
    
    def restore_mask(self, mask, ratio, pad, original_shape):

        h_ori, w_ori = original_shape[:2]

        # → 이미 원본 해상도라면 그대로 리턴
        if mask.shape[:2] == (h_ori, w_ori):
            return mask

        # 그렇지 않으면 letterbox 역연산
        pad_w, pad_h = int(pad[0]), int(pad[1])
        h_mask, w_mask = mask.shape[:2]

        # 1) padding 제거
        mask_cropped = mask[pad_h : h_mask - pad_h,
                            pad_w : w_mask - pad_w]

        # 2) 원본 크기로 리사이즈
        mask_resized = cv2.resize(
            mask_cropped,
            (w_ori, h_ori),                    # (width, height)
            interpolation=cv2.INTER_NEAREST
        )
        print(mask_resized.shape)

        return mask_resized

    def YolopDetection(self):

        while self.st_BGRImg is None:
            rospy.sleep(0.05)
            pass

        try:
            while not rospy.is_shutdown():
                image = self.st_BGRImg.copy()
                #cv2.imshow("YOLO Detection", image)

                input_tensor, ratio, pad = self.preprocess(image)

                with torch.no_grad():
                    [pred, anchor_grid], seg, ll = self.model(input_tensor)

                pred = split_for_trace_model(pred, anchor_grid)
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

                da_mask = torch.argmax(seg, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
                ll_mask = lane_line_mask(ll)

                # 1. 모델로부터 나온 마스크 (640x640)
                # 모델 예측 결과
                drivable_mask = (da_mask == 1).astype(np.uint8) * 255
                #lane_mask     = ll_mask.astype(np.uint8) * 255

                # letterbox 역변환으로 복원
                drivable_resized = self.restore_mask(drivable_mask, ratio, pad, image.shape)
                #lane_resized     = self.restore_mask(lane_mask, ratio, pad, image.shape)

                # 오버레이 생성
                overlay = np.zeros_like(image)
                #overlay[lane_resized > 127] = [255, 255, 255]   
                overlay[drivable_resized > 127] = [0, 255, 0]  

                # 합성
                result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

                # Bounding box 시각화
                # for det in pred:
                #     if det is not None and len(det):
                #         for *xyxy, conf, cls in det:
                #             xyxy = torch.tensor(xyxy).view(1, 4)
                #             xyxy = xyxy.cpu().numpy().astype(int).flatten()
                #             cv2.rectangle(result, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 255), 2)

                #print("mask shape:",da_mask.shape)
                cv2.imshow("YOLOPv2", overlay)
                cv2.imshow("YOLOPv2 Result", result)
                cv2.waitKey(1)
                self.rate.sleep()

        except Exception as e:
            rospy.logerr(f"[YOLOPv2 Node] 오류 발생: {e}")

if __name__ == "__main__":
    rospy.init_node('YOLOPv2', anonymous=True)
    YOLOPv2ROSNode()
    rospy.spin()
'''
    