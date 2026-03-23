#!/usr/bin/env python3
import os
import sys
import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import gc

# YOLOv8 라이브러리 추가
from ultralytics import YOLO

# yolopv2_code 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "../yolopv2_code"))
from utils.utils import letterbox

class DualSemanticMaskNode:
    def __init__(self):
        rospy.loginfo("듀얼 세그멘테이션 노드")

        # ===== 1. ROS 통신 설정 =====
        self.bridge = CvBridge()
        # 도로 영역 퍼블리셔
        self.road_pub = rospy.Publisher("/road_mask", Image, queue_size=1)
        # 장애물 영역 퍼블리셔
        self.obs_pub = rospy.Publisher("/obstacle_mask", Image, queue_size=1)
        
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback_Image, queue_size=1, buff_size=2**24)

        # ===== 2. 모델 로드 (GPU 우선) =====
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # (1) YOLOPv2 모델 로드
        yolop_path = rospy.get_param("~model_path", "../yolopv2_code/yolopv2.pt")
        self.yolop_model = torch.jit.load(yolop_path, map_location=self.device)
        self.yolop_model.eval()

        # (2) YOLOv8-seg 모델 로드
        rospy.loginfo("YOLOv8s-seg 모델 로딩 중...")
        self.yolo_seg_model = YOLO("yolov8s-seg.pt") # .to(self.device)는 모델 내부에서 자동 처리

        # ===== 3. 설정 파라미터 =====
        self.img_size = 640
        self.st_BGRImg = None
        self.rate = rospy.Rate(20) 

        rospy.on_shutdown(self.cleanup)
        self.main_loop()

    def cleanup(self):
        rospy.loginfo("노드를 종료합니다.")
        torch.cuda.empty_cache()
        gc.collect()
        cv2.destroyAllWindows()
    
    def callback_Image(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.st_BGRImg = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logerr(f"이미지 수신 오류: {e}")

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
            if self.st_BGRImg is None:
                rospy.sleep(0.05)
                continue

            image = self.st_BGRImg.copy()
            img_h, img_w = image.shape[:2]

            # ------------------------------------------
            # TRACK 1. YOLOPv2 (도로 영역)
            # ------------------------------------------
            input_tensor, ratio, pad = self.preprocess_yolop(image)
            with torch.no_grad():
                [pred, anchor_grid], seg, ll = self.yolop_model(input_tensor)
            
            da_mask = torch.argmax(seg, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
            road_raw = (da_mask == 1).astype(np.uint8) * 255
            road_mask_final = self.restore_mask(road_raw, ratio, pad, image.shape)

            # ------------------------------------------
            # TRACK 2. YOLOv8-seg (장애물 영역)
            # ------------------------------------------
            # YOLOv8은 내부적으로 전처리를 하므로 원본 이미지를 그대로 넣어도 됩니다.
            yolo_results = self.yolo_seg_model(image, verbose=False, device=self.device)
            obs_mask_final = np.zeros((img_h, img_w), dtype=np.uint8)

            if yolo_results[0].masks is not None:
                # 찾은 모든 객체의 세그멘테이션 폴리곤을 마스크에 그립니다.
                for poly in yolo_results[0].masks.xy:
                    pts = np.array(poly, np.int32)
                    cv2.fillPoly(obs_mask_final, [pts], 255)

            # ------------------------------------------
            # TRACK 3. 데이터 발행 및 시각화
            # ------------------------------------------
            # 시각용 오버레이 (도로: 초록, 장애물: 빨강)
            visual_img = image.copy()
            visual_img[road_mask_final > 127] = [0, 255, 0]
            visual_img[obs_mask_final > 127] = [0, 0, 255]
            
            # 실제 화면에 보여줄 때는 부드럽게 합성
            result_view = cv2.addWeighted(image, 0.6, visual_img, 0.4, 0)
            cv2.imshow("Dual Segmentation (Road & Obstacle)", result_view)

            # ROS 메시지 발행
            road_msg = self.bridge.cv2_to_imgmsg(road_mask_final, encoding="mono8")
            obs_msg = self.bridge.cv2_to_imgmsg(obs_mask_final, encoding="mono8")
            
            self.road_pub.publish(road_msg)
            self.obs_pub.publish(obs_msg)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            self.rate.sleep()

if __name__ == "__main__":
    rospy.init_node('dual_semantic_mask_node', anonymous=True)
    try:
        DualSemanticMaskNode()
    except rospy.ROSInterruptException:
        pass


'''
#!/usr/bin/env python3
import os
import sys
import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import gc
from ultralytics import YOLO

# Qt 경고 숨기기
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

sys.path.append(os.path.join(os.path.dirname(__file__), "../yolopv2_code"))
from utils.utils import letterbox

class DualSemanticMaskNode:
    def __init__(self):
        rospy.loginfo("road & obstacle mask")

        # ===== 1. 상태 변수 초기화  =====
        self.bridge = CvBridge()
        self.st_BGRImg = None
        self.img_size = 640
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_processing = False 
        self.frame_count = 0

        # ===== 2. 모델 로드 (Half 모드 제외하여 dtype 에러 차단) =====
        self.yolop_model = torch.jit.load(rospy.get_param("~model_path", "../yolopv2_code/yolopv2.pt"), map_location=self.device)
        self.yolop_model.eval()

        self.yolo_seg_model = YOLO("yolov8s-seg.pt")
        if self.device.type == 'cuda':
            self.yolo_seg_model.to(self.device) # .half()를 일단 제거하여 안정성 확보

        # ===== 3. ROS 통신 =====
        self.road_pub = rospy.Publisher("/road_mask", Image, queue_size=1)
        self.obs_pub = rospy.Publisher("/obstacle_mask", Image, queue_size=1)
        
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback_Image, queue_size=1, buff_size=2**24)

        rospy.on_shutdown(self.cleanup)
        self.main_loop()

    def cleanup(self):
        torch.cuda.empty_cache()
        gc.collect()
        cv2.destroyAllWindows()
    
    def callback_Image(self, msg):
        if self.is_processing:
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.st_BGRImg = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except:
            pass

    def main_loop(self):
        while not rospy.is_shutdown():
            if self.st_BGRImg is None:
                rospy.sleep(0.001)
                continue

            self.is_processing = True
            image = self.st_BGRImg.copy()
            self.st_BGRImg = None 
            
            img_h, img_w = image.shape[:2]

            # TRACK 1. YOLOPv2
            input_tensor, ratio, pad = self.preprocess_yolop(image)
            with torch.no_grad():
                _, seg, _ = self.yolop_model(input_tensor)
            
            da_mask = torch.argmax(seg, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            road_mask_final = self.restore_mask((da_mask == 1).astype(np.uint8) * 255, ratio, pad, image.shape)

            # TRACK 2. YOLOv8-seg (dtype 에러 방지를 위해 half=False)
            yolo_results = self.yolo_seg_model(image, verbose=False, device=self.device, half=False)
            obs_mask_final = np.zeros((img_h, img_w), dtype=np.uint8)

            if yolo_results[0].masks is not None:
                for poly in yolo_results[0].masks.xy:
                    cv2.fillPoly(obs_mask_final, [np.array(poly, np.int32)], 255)

            if self.frame_count % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            # 시각화 및 발행
            visual_img = image.copy()
            visual_img[road_mask_final > 127] = [0, 255, 0]
            visual_img[obs_mask_final > 127] = [0, 0, 255]
            
            cv2.imshow("Dual Track", cv2.resize(cv2.addWeighted(image, 0.6, visual_img, 0.4, 0), (640, 480)))
            
            self.road_pub.publish(self.bridge.cv2_to_imgmsg(road_mask_final, encoding="mono8"))
            self.obs_pub.publish(self.bridge.cv2_to_imgmsg(obs_mask_final, encoding="mono8"))

            self.is_processing = False # 처리가 끝났음을 알림
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def preprocess_yolop(self, img):
        resized_img, ratio, pad = letterbox(img, new_shape=self.img_size, stride=32)
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().to(self.device) / 255.0
        return img_tensor.unsqueeze(0), ratio, pad

    def restore_mask(self, mask, ratio, pad, original_shape):
        h_ori, w_ori = original_shape[:2]
        pad_w, pad_h = int(pad[0]), int(pad[1])
        mask_cropped = mask[pad_h : mask.shape[0] - pad_h, pad_w : mask.shape[1] - pad_w]
        return cv2.resize(mask_cropped, (w_ori, h_ori), interpolation=cv2.INTER_NEAREST)

if __name__ == "__main__":
    rospy.init_node('dual_semantic_mask_node', anonymous=True)
    DualSemanticMaskNode()
    '''