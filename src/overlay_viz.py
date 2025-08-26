import rclpy, cv2
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import csv, json
from my_msgs.msg import BoundingBoxes, BoundingBox
from my_msgs.msg import TrackArray

def _color_for(name: str):
    name = (name or '').lower()
    if name in ('car', 'vehicle'):      return (0, 255, 0)     # green
    if name in ('person', 'pedestrian'): return (255, 0, 0)    # blue (BGR)
    if name in ('traffic light','traffic_light','signal'): return (0, 0, 255) # red
    return (0, 255, 255)  # default: yellow

try:
    from bbox_msgs.msg import BoundingBoxes, BoundingBox
except ImportError:
    from my_msgs.msg import BoundingBoxes, BoundingBox

class Overlay(Node):
    def __init__(self):
        super().__init__('overlay_viz')
        self.bridge = CvBridge()
        self.image_topic = self.declare_parameter('image_topic','/image_raw').get_parameter_value().string_value
        self.boxes_topic = self.declare_parameter('boxes_topic','/yolo/bounding_boxes').get_parameter_value().string_value
        self.out_topic   = self.declare_parameter('out_topic','/image_yolo').get_parameter_value().string_value

        # Overlay / filter params
        self.conf_min = self.declare_parameter('overlay_conf_min', 0.25).get_parameter_value().double_value
        self.show_fps  = self.declare_parameter('show_fps', True).get_parameter_value().bool_value

        # Saving options
        self.save_mp4   = self.declare_parameter('save_mp4', False).get_parameter_value().bool_value
        self.mp4_path   = self.declare_parameter('mp4_path', '/workspace/out.mp4').get_parameter_value().string_value
        self.mp4_fps    = int(self.declare_parameter('mp4_fps', 20).get_parameter_value().integer_value)
        self.save_images= self.declare_parameter('save_images', False).get_parameter_value().bool_value
        self.images_dir = self.declare_parameter('images_dir', '/workspace/frames').get_parameter_value().string_value
        self.images_every_n = int(self.declare_parameter('images_every_n', 1).get_parameter_value().integer_value)

        self.save_boxes_csv = self.declare_parameter('save_boxes_csv', False).get_parameter_value().bool_value
        self.csv_path = self.declare_parameter('csv_path', '/workspace/out_boxes.csv').get_parameter_value().string_value
        self.save_boxes_json = self.declare_parameter('save_boxes_json', False).get_parameter_value().bool_value
        self.json_path = self.declare_parameter('json_path', '/workspace/out_boxes.json').get_parameter_value().string_value

        self.sub_tracks = self.create_subscription(TrackArray, '/tracks', self.cb_tracks, qos_profile_sensor_data)
        self.last_tracks = []

        self._csv_file = None
        self._csv_writer = None
        if self.save_boxes_csv:
            self._csv_file = open(self.csv_path, 'w', newline='')
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(['frame','class','prob','xmin','ymin','xmax','ymax'])
        if self.save_boxes_json:
            self._json_records = []

        # Runtime state
        self._fps_ema = None
        self._last_stamp = None
        self._frame_idx = 0
        self._writer = None
        if self.save_images:
            os.makedirs(self.images_dir, exist_ok=True)

        self.qos_reliable = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        self.sub_img_be = self.create_subscription(Image, self.image_topic, self.cb_img, qos_profile_sensor_data)
        self.sub_img_rel = self.create_subscription(Image, self.image_topic, self.cb_img, self.qos_reliable)
        self.sub_box = self.create_subscription(BoundingBoxes, self.boxes_topic, self.cb_boxes, qos_profile_sensor_data)
        self.pub_img = self.create_publisher(Image, self.out_topic, self.qos_reliable)

        self.last_img = None
        self.last_boxes = []
        self.pub_n = 0
        self.get_logger().info(f'overlay_viz: img={self.image_topic}, boxes={self.boxes_topic}, out={self.out_topic}')
        self.get_logger().info('Hint: if no output, check that /image_raw and /yolo/bounding_boxes are publishing and QoS matches (SensorData).')

    def cb_img(self, msg: Image):
        self.get_logger().debug(f'recv image: {msg.width}x{msg.height} ts={msg.header.stamp.sec}.{msg.header.stamp.nanosec}')
        self.last_img = msg
        self.try_publish()

    def cb_boxes(self, msg: BoundingBoxes):
        self.get_logger().debug(f'recv boxes: {len(msg.boxes)}')
        # cache for draw step
        try:
            self.last_boxes = list(msg.boxes)
        except Exception:
            self.last_boxes = []
        self.try_publish()

    def cb_tracks(self, msg: TrackArray):
        self.get_logger().debug(f'recv tracks: {len(msg.tracks)}')
        try:
            self.last_tracks = list(msg.tracks)
        except Exception:
            self.last_tracks = []
        self.try_publish()

    def try_publish(self):
        if self.last_img is None:
            return
        img = self.bridge.imgmsg_to_cv2(self.last_img, desired_encoding='bgr8')
        h, w = img.shape[:2]

        # ==== draw boxes with class colors and threshold ====
        for b in self.last_boxes:
            name = getattr(b, 'class_id', '')
            conf = float(getattr(b, 'probability', 0.0))
            if conf < self.conf_min:
                continue
            x1, y1, x2, y2 = int(b.xmin), int(b.ymin), int(b.xmax), int(b.ymax)
            x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
            y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
            color = _color_for(name)
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            label = f'{name} {conf:.2f}' if name else f'{conf:.2f}'
            cv2.putText(img, label, (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # ==== draw tracks (ID overlay) ====
        if hasattr(self, 'last_tracks') and self.last_tracks:
            for t in self.last_tracks:
                name = getattr(t, 'class_id', '')
                score = float(getattr(t, 'score', 0.0))
                x1, y1, x2, y2 = int(t.xmin), int(t.ymin), int(t.xmax), int(t.ymax)
                x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
                y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
                color = _color_for(name)  # same color map as detections
                # thicker rectangle for tracks to distinguish
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    img,
                    f"{name}#{int(getattr(t,'id',-1))} {score:.2f}",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        # ==== save raw bounding boxes ====
        if self.save_boxes_csv and self._csv_writer is not None:
            for b in self.last_boxes:
                self._csv_writer.writerow([self._frame_idx, getattr(b,'class_id',''), float(getattr(b,'probability',0.0)), int(b.xmin), int(b.ymin), int(b.xmax), int(b.ymax)])
        if self.save_boxes_json:
            frame_record = []
            for b in self.last_boxes:
                frame_record.append({
                    'frame': self._frame_idx,
                    'class': getattr(b,'class_id',''),
                    'prob': float(getattr(b,'probability',0.0)),
                    'xmin': int(b.xmin),
                    'ymin': int(b.ymin),
                    'xmax': int(b.xmax),
                    'ymax': int(b.ymax),
                })
            self._json_records.append(frame_record)

        # ==== FPS overlay ====
        if self.show_fps:
            now = self.get_clock().now()
            if self._last_stamp is not None:
                dt = (now - self._last_stamp).nanoseconds / 1e9
                if dt > 0:
                    inst = 1.0 / dt
                    self._fps_ema = inst if self._fps_ema is None else (0.1*inst + 0.9*self._fps_ema)
            self._last_stamp = now
            if self._fps_ema is not None:
                cv2.putText(img, f'FPS: {self._fps_ema:.1f}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

        # ==== save outputs ====
        self._frame_idx += 1
        if self.save_mp4 and self._writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._writer = cv2.VideoWriter(self.mp4_path, fourcc, float(self.mp4_fps), (w, h))
            if not self._writer.isOpened():
                self.get_logger().warn(f'VideoWriter open failed: {self.mp4_path}')
                self._writer = None
        if self._writer is not None:
            try:
                self._writer.write(img)
            except Exception as e:
                self.get_logger().warn(f'VideoWriter write failed: {e}')
        if self.save_images and (self._frame_idx % max(1, self.images_every_n) == 0):
            try:
                cv2.imwrite(os.path.join(self.images_dir, f'frame_{self._frame_idx:06d}.jpg'), img)
            except Exception as e:
                self.get_logger().warn(f'Image save failed: {e}')

        out = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        # 이미지 헤더 복사(타임싱크/리플레이 유용)
        out.header = self.last_img.header
        self.pub_img.publish(out)
        self.pub_n += 1
        if self.pub_n % 30 == 0:
            self.get_logger().info(f'published #{self.pub_n} to {self.out_topic} - {img.shape[1]}x{img.shape[0]}, boxes={len(self.last_boxes)}')

    def destroy_node(self):
        try:
            if self._writer is not None:
                self._writer.release()
                self.get_logger().info(f'MP4 saved to {self.mp4_path}')
        except Exception:
            pass
        try:
            if self._csv_file is not None:
                self._csv_file.close()
                self.get_logger().info(f'CSV saved to {self.csv_path}')
        except Exception:
            pass
        try:
            if self.save_boxes_json:
                with open(self.json_path,'w') as f:
                    json.dump(self._json_records,f,indent=2)
                self.get_logger().info(f'JSON saved to {self.json_path}')
        except Exception:
            pass
        super().destroy_node()

def main():
    rclpy.init()
    rclpy.spin(Overlay())
    rclpy.shutdown()

if __name__ == '__main__':
    main()