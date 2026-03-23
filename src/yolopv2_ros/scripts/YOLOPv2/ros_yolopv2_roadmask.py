#!/usr/bin/env python3
# ros_yolopv2_roadmask.py
# Subscribe MORAI compressed image topic, run YOLOPv2 (TorchScript), publish road mask + overlay.

import os
import rospy
import cv2
import numpy as np
import torch

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

# YOLOPv2 utils (these imports match the official YOLOPv2 repo structure)
from utils.utils import letterbox
from utils.utils import driving_area_mask  # demo.py uses this to get drivable-area mask


class YOLOPv2RoadMaskNode:
    def __init__(self):
        self.bridge = CvBridge()

        # Params
        self.image_topic = rospy.get_param("~image_topic", "/image_jpeg/compressed")
        self.weights = rospy.get_param("~weights", "data/weights/yolopv2.pt")
        self.device = rospy.get_param("~device", "cpu")  # "cpu" or GPU index string like "0"
        self.img_size = int(rospy.get_param("~img_size", 640))
        self.mask_thresh = float(rospy.get_param("~mask_thresh", 0.5))

        # Device
        if str(self.device).lower() == "cpu":
            self.torch_device = torch.device("cpu")
        else:
            self.torch_device = torch.device(f"cuda:{self.device}")

        # Load TorchScript weights
        if not os.path.exists(self.weights):
            raise FileNotFoundError(f"weights not found: {self.weights}")
        self.model = torch.jit.load(self.weights, map_location=self.torch_device)
        self.model.eval()

        # Publishers
        self.pub_mask = rospy.Publisher("/yolopv2/road_mask", Image, queue_size=1)
        self.pub_overlay = rospy.Publisher("/yolopv2/overlay", Image, queue_size=1)

        # Subscriber: CompressedImage
        rospy.Subscriber(
            self.image_topic, CompressedImage, self.cb_compressed,
            queue_size=1, buff_size=2**24
        )

        rospy.loginfo(f"[YOLOPv2] subscribed: {self.image_topic}")
        rospy.loginfo("[YOLOPv2] publishing: /yolopv2/road_mask, /yolopv2/overlay")
        rospy.loginfo(f"[YOLOPv2] device: {self.device}, weights: {self.weights}")

    def cb_compressed(self, msg: CompressedImage):
        # Decode JPEG bytes -> BGR image
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr is None:
                rospy.logwarn("cv2.imdecode returned None (bad jpeg?)")
                return
        except Exception as e:
            rospy.logwarn(f"CompressedImage decode failed: {e}")
            return

        img0 = bgr  # BGR HxWx3

        # Preprocess (letterbox to img_size)
        img = letterbox(img0, self.img_size, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img)

        im = torch.from_numpy(img).to(self.torch_device)
        im = im.float() / 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        # Inference
        with torch.no_grad():
            out = self.model(im)

        # YOLOPv2 TorchScript usually returns (pred, seg, ll)
        # We'll robustly handle list/tuple outputs
        if isinstance(out, (list, tuple)) and len(out) >= 3:
            pred, seg, ll = out[0], out[1], out[2]
        else:
            rospy.logwarn("Unexpected model output format. Check YOLOPv2 demo.py for output structure.")
            return

        # Drivable-area mask (prob or logits -> mask via helper)
        da = driving_area_mask(seg)  # expected HxW (or 1xHxW)
        if hasattr(da, "detach"):
            da = da.squeeze().detach().cpu().numpy()
        else:
            da = np.squeeze(da)

        road = (da > self.mask_thresh).astype(np.uint8) * 255  # 0/255
        h0, w0 = img0.shape[:2]
        if road.shape[0] != h0 or road.shape[1] != w0:
            road = cv2.resize(road, (w0, h0), interpolation=cv2.INTER_NEAREST)


        # Overlay for visualization
        overlay = img0.copy()
        # Paint road pixels green (simple alpha blend)
        green = np.array([0, 255, 0], dtype=np.uint8)
        idx = road == 255
        overlay[idx] = (0.6 * overlay[idx] + 0.4 * green).astype(np.uint8)

        # Publish mask + overlay
        mask_msg = self.bridge.cv2_to_imgmsg(road, encoding="mono8")
        mask_msg.header = msg.header
        self.pub_mask.publish(mask_msg)

        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        overlay_msg.header = msg.header
        self.pub_overlay.publish(overlay_msg)


def main():
    rospy.init_node("yolopv2_roadmask_node", anonymous=False)
    YOLOPv2RoadMaskNode()
    rospy.spin()


if __name__ == "__main__":
    main()