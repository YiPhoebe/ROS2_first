import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
from ultralytics import YOLO

from rcl_interfaces.msg import SetParametersResult

from my_msgs.msg import BoundingBox, BoundingBoxes

import numpy as np
import cv2
import tempfile
import os
import random
try:
    import torch
except Exception:
    torch = None

def _color_for(name: str):
    name = (name or '').lower()
    if name in ('car', 'vehicle'):      return (0, 255, 0)     # green
    if name in ('person', 'pedestrian'): return (255, 0, 0)    # blue
    if name in ('traffic light','traffic_light','signal'): return (0, 0, 255) # red
    return (0, 255, 255)  # default: yellow

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_subscriber_py_node')
        self.bridge = CvBridge()

        weights = self.declare_parameter('weights', '/workspace/yolo11n.pt').get_parameter_value().string_value
        self.conf_thres = self.declare_parameter('conf', 0.03).get_parameter_value().double_value
        # Declare additional tunable parameters
        self.imgsz = int(self.declare_parameter('imgsz', 480).get_parameter_value().integer_value)
        self.every_n = int(self.declare_parameter('every_n', 1).get_parameter_value().integer_value)
        self.sub_topic = self.declare_parameter('sub_topic', '/image_raw').get_parameter_value().string_value
        self.pub_topic = self.declare_parameter('pub_topic', '/yolo/bounding_boxes').get_parameter_value().string_value
        self.debug_log = self.declare_parameter('debug_log', False).get_parameter_value().bool_value
        self.log_detections = self.declare_parameter('log_detections', False).get_parameter_value().bool_value

        # Overlay/filter params
        self.conf_min = self.declare_parameter('overlay_conf_min', 0.25).get_parameter_value().double_value
        self.show_fps  = self.declare_parameter('show_fps', True).get_parameter_value().bool_value

        # Saving options
        self.save_mp4   = self.declare_parameter('save_mp4', False).get_parameter_value().bool_value
        self.mp4_path   = self.declare_parameter('mp4_path', '/workspace/out.mp4').get_parameter_value().string_value
        self.mp4_fps    = int(self.declare_parameter('mp4_fps', 20).get_parameter_value().integer_value)
        self.save_images= self.declare_parameter('save_images', False).get_parameter_value().bool_value
        self.images_dir = self.declare_parameter('images_dir', '/workspace/frames').get_parameter_value().string_value
        self.images_every_n = int(self.declare_parameter('images_every_n', 1).get_parameter_value().integer_value)

        # Runtime state
        self._last_img_size = None
        self._fps_ema = None
        self._frame_idx = 0
        self._writer = None
        if self.save_images:
            os.makedirs(self.images_dir, exist_ok=True)

        self.use_tempfile = self.declare_parameter('use_tempfile', False).get_parameter_value().bool_value

        # Reproducibility seed (best-effort)
        self.seed = int(self.declare_parameter('seed', 42).get_parameter_value().integer_value)
        try:
            random.seed(self.seed)
            np.random.seed(self.seed)
            if torch is not None:
                torch.manual_seed(self.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.seed)
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    pass
                try:
                    import torch.backends.cudnn as cudnn
                    cudnn.deterministic = True
                    cudnn.benchmark = False
                except Exception:
                    pass
        except Exception:
            pass

        # Internal counters
        self._frame_count = 0

        # Allow runtime updates via ros2 param set
        self.add_on_set_parameters_callback(self._on_params)
        self.model = YOLO(weights)

        qos = qos_profile_sensor_data
        self.sub = self.create_subscription(Image, self.sub_topic, self.cb_image, qos)
        self.pub = self.create_publisher(BoundingBoxes, self.pub_topic, 10)

        self.get_logger().info(
            f'YOLO ready (weights={weights}, conf={self.conf_thres}, imgsz={self.imgsz}, every_n={self.every_n}, sub={self.sub_topic}, pub={self.pub_topic})'
        )

    def cb_image(self, msg: Image):
        # Frame skipping based on every_n
        self._frame_count += 1
        if self.every_n > 1 and (self._frame_count % self.every_n) != 0:
            return
        try:
            # ROS Image -> OpenCV BGR (keep BGR; send directly to Ultralytics predict like CLI)
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        # ==== FPS overlay ====
        if self.show_fps:
            now = self.get_clock().now()
            if self._last_img_size is None:
                self._last_img_size = (img_bgr.shape[1], img_bgr.shape[0])
            if hasattr(self, '_last_stamp'):
                dt = (now - self._last_stamp).nanoseconds / 1e9 if (now and self._last_stamp) else 0.0
                if dt > 0:
                    inst_fps = 1.0 / dt
                    self._fps_ema = inst_fps if self._fps_ema is None else (0.1*inst_fps + 0.9*self._fps_ema)
            self._last_stamp = now
            if self._fps_ema is not None:
                cv2.putText(img_bgr, f"FPS: {self._fps_ema:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

        # ==== save outputs ====
        self._frame_idx += 1
        # lazy-init writer (mp4)
        if self.save_mp4 and self._writer is None:
            h, w = img_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._writer = cv2.VideoWriter(self.mp4_path, fourcc, float(self.mp4_fps), (w, h))
            if not self._writer.isOpened():
                self.get_logger().warn(f'VideoWriter open failed: {self.mp4_path}')
                self._writer = None
        if self._writer is not None:
            try:
                self._writer.write(img_bgr)
            except Exception as e:
                self.get_logger().warn(f'VideoWriter write failed: {e}')
        if self.save_images and (self._frame_idx % max(1, self.images_every_n) == 0):
            try:
                out_path = os.path.join(self.images_dir, f'frame_{self._frame_idx:06d}.jpg')
                cv2.imwrite(out_path, img_bgr)
            except Exception as e:
                self.get_logger().warn(f'Image save failed: {e}')

        # Optional debug: frame stats
        if self.debug_log:
            try:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                avg_brightness = gray.mean()
                h, w = img_bgr.shape[:2]
                self.get_logger().debug(f'frame: {w}x{h}, avg_brightness={avg_brightness:.2f}, dtype={img_bgr.dtype}')
            except Exception:
                pass

        # Run single inference (BGR ndarray by default, optional tempfile path)
        r_use = None
        try:
            if self.use_tempfile:
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as tmpfile:
                    cv2.imwrite(tmpfile.name, img_bgr)
                    results = self.model.predict(source=tmpfile.name, conf=self.conf_thres, imgsz=self.imgsz, verbose=False)
            else:
                results = self.model.predict(source=img_bgr, conf=self.conf_thres, imgsz=self.imgsz, verbose=False)
            if results:
                r_use = results[0]
        except Exception as e:
            self.get_logger().warn(f'inference error: {e}')

        # Build and publish message
        out = BoundingBoxes()
        if hasattr(out, 'header'):
            out.header = Header()
            out.header.stamp = msg.header.stamp if hasattr(msg, 'header') else self.get_clock().now().to_msg()
            out.header.frame_id = msg.header.frame_id if hasattr(msg, 'header') else 'camera'

        boxes = []
        if r_use is not None and getattr(r_use, 'boxes', None) is not None:
            names_map = getattr(self.model, 'names', None)
            for b in r_use.boxes:
                xyxy = b.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                cls_id = int(b.cls[0]) if b.cls is not None else -1
                conf = float(b.conf[0]) if b.conf is not None else 0.0
                name = names_map.get(cls_id, str(cls_id)) if names_map else str(cls_id)
                if self.log_detections:
                    self.get_logger().info(f'Detection: name={name}, conf={conf:.3f}, bbox=({x1}, {y1}, {x2}, {y2})')

                bb = BoundingBox()
                if hasattr(bb, 'x'): bb.x = x1
                if hasattr(bb, 'y'): bb.y = y1
                if hasattr(bb, 'width'): bb.width = w
                if hasattr(bb, 'height'): bb.height = h
                if hasattr(bb, 'xmin'): bb.xmin = x1
                if hasattr(bb, 'ymin'): bb.ymin = y1
                if hasattr(bb, 'xmax'): bb.xmax = x2
                if hasattr(bb, 'ymax'): bb.ymax = y2
                if hasattr(bb, 'class_id'): bb.class_id = name
                if hasattr(bb, 'label'): bb.label = name
                if hasattr(bb, 'probability'): bb.probability = conf
                if hasattr(bb, 'score'): bb.score = conf
                if hasattr(bb, 'id'): bb.id = cls_id
                boxes.append(bb)

        if hasattr(out, 'boxes'):
            out.boxes = boxes
        elif hasattr(out, 'bounding_boxes'):
            out.bounding_boxes = boxes

        self.pub.publish(out)

    def _on_params(self, params):
        # Accept dynamic updates to parameters
        for p in params:
            if p.name == 'conf':
                try:
                    self.conf_thres = float(p.value)
                except Exception:
                    pass
            elif p.name == 'imgsz':
                try:
                    self.imgsz = int(p.value)
                except Exception:
                    pass
            elif p.name == 'every_n':
                try:
                    self.every_n = int(p.value)
                except Exception:
                    pass
            elif p.name == 'sub_topic':
                try:
                    self.sub_topic = str(p.value)
                except Exception:
                    pass
            elif p.name == 'pub_topic':
                try:
                    self.pub_topic = str(p.value)
                except Exception:
                    pass
            elif p.name == 'use_tempfile':
                try:
                    self.use_tempfile = bool(p.value)
                except Exception:
                    pass
            elif p.name == 'debug_log':
                try:
                    self.debug_log = bool(p.value)
                except Exception:
                    pass
            elif p.name == 'log_detections':
                try:
                    self.log_detections = bool(p.value)
                except Exception:
                    pass
        return SetParametersResult(successful=True)

    def destroy_node(self):
        try:
            if getattr(self, '_writer', None) is not None:
                self._writer.release()
                self.get_logger().info(f'MP4 saved to {self.mp4_path}')
        except Exception:
            pass
        super().destroy_node()

def main():
    rclpy.init()
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
