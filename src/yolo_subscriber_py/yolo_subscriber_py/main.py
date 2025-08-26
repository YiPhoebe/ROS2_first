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
        self._last_boxes = []
        self._last_img_size = None
        self._fps_ema = None
        self._frame_idx = 0
        self._writer = None
        if self.save_images:
            os.makedirs(self.images_dir, exist_ok=True)

        self.use_tempfile = self.declare_parameter('use_tempfile', False).get_parameter_value().bool_value

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

    def cb_boxes(self, msg: BoundingBoxes):
        try:
            self._last_boxes = list(msg.boxes) if hasattr(msg, 'boxes') else list(getattr(msg, 'bounding_boxes', []))
        except Exception:
            self._last_boxes = []
        self.get_logger().debug(f'recv boxes: {len(self._last_boxes)}')

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

        # ==== draw boxes (class-colored) ====
        boxes = getattr(self, '_last_boxes', []) or []
        for bb in boxes:
            try:
                # support either xmin/xmax or x/width style fields
                if hasattr(bb, 'xmin'):
                    x1, y1, x2, y2 = int(bb.xmin), int(bb.ymin), int(bb.xmax), int(bb.ymax)
                else:
                    x1, y1 = int(getattr(bb, 'x', 0)), int(getattr(bb, 'y', 0))
                    x2 = x1 + int(getattr(bb, 'width', 0))
                    y2 = y1 + int(getattr(bb, 'height', 0))
                name = getattr(bb, 'class_id', None) or getattr(bb, 'label', '')
                conf = float(getattr(bb, 'probability', getattr(bb, 'score', 0.0)))
                if conf < self.conf_min:
                    continue
                color = _color_for(name)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                label = f"{name} {conf:.2f}" if name else f"{conf:.2f}"
                cv2.putText(img_bgr, label, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            except Exception:
                continue

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

        # Ensure RGB format and contiguous memory for YOLO
        try:
            frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            frame_rgb = img_bgr[..., ::-1]
        frame_rgb = np.ascontiguousarray(frame_rgb)

        # Compute average brightness
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        avg_brightness = gray.mean()
        self.get_logger().info(f'Average brightness: {avg_brightness:.2f}')

        # Log frame info
        try:
            h, w = img_bgr.shape[:2]
            self.get_logger().info(f'frame: shape=({h},{w},3), dtype={img_bgr.dtype}')
        except Exception:
            pass

        # Run inference using three sources and select best result
        results_list = []

        # 1. RGB ndarray
        results_rgb = self.model.predict(source=frame_rgb, conf=self.conf_thres, imgsz=self.imgsz, verbose=False)
        det_count_rgb = 0
        if results_rgb:
            r0_rgb = results_rgb[0]
            if hasattr(r0_rgb, 'boxes') and r0_rgb.boxes is not None:
                det_count_rgb = len(r0_rgb.boxes)
        results_list.append((det_count_rgb, results_rgb[0] if results_rgb else None))

        # 2. BGR ndarray
        results_bgr = self.model.predict(source=img_bgr, conf=self.conf_thres, imgsz=self.imgsz, verbose=False)
        det_count_bgr = 0
        if results_bgr:
            r0_bgr = results_bgr[0]
            if hasattr(r0_bgr, 'boxes') and r0_bgr.boxes is not None:
                det_count_bgr = len(r0_bgr.boxes)
        results_list.append((det_count_bgr, results_bgr[0] if results_bgr else None))

        # 3. Optional: Temporary JPEG file saved from frame_bgr
        r_temp = None
        det_count_temp = 0
        if self.use_tempfile:
            try:
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as tmpfile:
                    cv2.imwrite(tmpfile.name, img_bgr)
                    results_temp = self.model.predict(source=tmpfile.name, conf=self.conf_thres, imgsz=self.imgsz, verbose=False)
                    if results_temp:
                        r0_temp = results_temp[0]
                        if hasattr(r0_temp, 'boxes') and r0_temp.boxes is not None:
                            det_count_temp = len(r0_temp.boxes)
                        r_temp = r0_temp
            except Exception as e:
                self.get_logger().warn(f'Error during temporary file inference: {e}')
        results_list.append((det_count_temp, r_temp))

        # Select best result
        det_n, r_use = max(results_list, key=lambda x: x[0]) if results_list else (0, None)

        self.get_logger().info(f'Detections counts - RGB: {det_count_rgb}, BGR: {det_count_bgr}, Temp JPEG: {det_count_temp}')

        # Build and publish message
        out = BoundingBoxes()
        if hasattr(out, 'header'):
            out.header = Header()
            out.header.stamp = msg.header.stamp if hasattr(msg, 'header') else self.get_clock().now().to_msg()
            out.header.frame_id = msg.header.frame_id if hasattr(msg, 'header') else 'camera'

        boxes = []
        if r_use is not None and det_n > 0:
            names_map = getattr(self.model, 'names', None)
            for b in r_use.boxes:
                xyxy = b.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                cls_id = int(b.cls[0]) if b.cls is not None else -1
                conf = float(b.conf[0]) if b.conf is not None else 0.0
                name = names_map.get(cls_id, str(cls_id)) if names_map else str(cls_id)

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
