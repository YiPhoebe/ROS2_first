import rclpy, cv2
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import csv, json
from my_msgs.msg import TrackArray

def _color_for(name: str):
    name = (name or '').lower()
    if name in ('car', 'vehicle'):      return (0, 255, 0)     # green
    if name in ('person', 'pedestrian'): return (255, 0, 0)    # blue (BGR)
    if name in ('traffic light','traffic_light','signal'): return (0, 0, 255) # red
    return (0, 255, 255)  # default: yellow

try:
    from my_msgs.msg import BoundingBoxes, BoundingBox
except ImportError:
    from bbox_msgs.msg import BoundingBoxes, BoundingBox

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
        # Timestamp sync tolerance (ms) between image and boxes/tracks
        self.sync_tol_ms = float(self.declare_parameter('sync_tol_ms', 50.0).get_parameter_value().double_value)
        self.buffer_size = int(self.declare_parameter('sync_buffer_size', 200).get_parameter_value().integer_value)
        # Optionally reuse last detections to fill gaps between detector frames
        self.hold_last_boxes = self.declare_parameter('hold_last_boxes', True).get_parameter_value().bool_value
        # When true, also hold when detections exist but all fall below overlay_conf_min
        self.hold_on_empty = self.declare_parameter('hold_on_empty', True).get_parameter_value().bool_value
        self.hold_last_ms = float(self.declare_parameter('hold_last_ms', 800.0).get_parameter_value().double_value)

        # Saving options
        self.save_mp4   = self.declare_parameter('save_mp4', False).get_parameter_value().bool_value
        self.mp4_path   = self.declare_parameter('mp4_path', '/workspace/out.mp4').get_parameter_value().string_value
        self.mp4_fps    = int(self.declare_parameter('mp4_fps', 20).get_parameter_value().integer_value)
        self.save_images= self.declare_parameter('save_images', False).get_parameter_value().bool_value
        self.images_dir = self.declare_parameter('images_dir', '/workspace/frames').get_parameter_value().string_value
        self.images_every_n = int(self.declare_parameter('images_every_n', 1).get_parameter_value().integer_value)
        # Save only when at least this many detections (after conf filter)
        self.save_min_boxes = int(self.declare_parameter('save_min_boxes', 0).get_parameter_value().integer_value)

        self.save_boxes_csv = self.declare_parameter('save_boxes_csv', False).get_parameter_value().bool_value
        self.csv_path = self.declare_parameter('csv_path', '/workspace/out_boxes.csv').get_parameter_value().string_value
        self.save_boxes_json = self.declare_parameter('save_boxes_json', False).get_parameter_value().bool_value
        self.json_path = self.declare_parameter('json_path', '/workspace/out_boxes.json').get_parameter_value().string_value

        self.sub_tracks = self.create_subscription(TrackArray, '/tracks', self.cb_tracks, qos_profile_sensor_data)
        self.tracks_by_stamp = {}

        self._csv_file = None
        self._csv_writer = None
        if self.save_boxes_csv:
            # Ensure output directory exists
            csv_dir = os.path.dirname(self.csv_path) or '.'
            os.makedirs(csv_dir, exist_ok=True)
            self._csv_file = open(self.csv_path, 'w', newline='')
            self._csv_writer = csv.writer(self._csv_file)
            # Include both local frame index (0-based) and header stamp for alignment
            self._csv_writer.writerow(['frame','frame_idx','stamp_sec','stamp_nanosec','class','prob','xmin','ymin','xmax','ymax'])
        if self.save_boxes_json:
            # Ensure output directory exists
            json_dir = os.path.dirname(self.json_path) or '.'
            os.makedirs(json_dir, exist_ok=True)
            self._json_records = []

        # Runtime state
        self._fps_ema = None
        self._last_stamp = None
        self._frame_idx = 0
        self._first_write_idx = None  # first frame index that actually produced a record
        self._writer = None
        self._last_img_stamp_ns = None  # dedup images arriving via multiple QoS subscribers
        self._last_img_recv_ns = None   # arrival-time based dedup fallback
        if self.save_images:
            os.makedirs(self.images_dir, exist_ok=True)

        self.qos_reliable = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        self.sub_img_be = self.create_subscription(Image, self.image_topic, self.cb_img, qos_profile_sensor_data)
        self.sub_img_rel = self.create_subscription(Image, self.image_topic, self.cb_img, self.qos_reliable)
        # Subscribe to boxes with both SensorData (BEST_EFFORT) and RELIABLE QoS to maximize compatibility
        self.sub_box = self.create_subscription(BoundingBoxes, self.boxes_topic, self.cb_boxes, qos_profile_sensor_data)
        self.sub_box_rel = self.create_subscription(BoundingBoxes, self.boxes_topic, self.cb_boxes, self.qos_reliable)
        self.pub_img = self.create_publisher(Image, self.out_topic, self.qos_reliable)

        self.last_img = None
        self.images_by_stamp = {}
        self.boxes_by_stamp = {}
        self._last_boxes = None
        self._last_boxes_stamp_ns = None
        self._last_vis_boxes = None
        self.pub_n = 0
        self.get_logger().info(f'overlay_viz: img={self.image_topic}, boxes={self.boxes_topic}, out={self.out_topic}')
        self.get_logger().info('Hint: if no output, check that /image_raw and /yolo/bounding_boxes are publishing and QoS matches (SensorData).')

    def cb_img(self, msg: Image):
        # dedup: same timestamp image may arrive twice (BEST_EFFORT + RELIABLE)
        now_ns = self.get_clock().now().nanoseconds
        try:
            stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        except Exception:
            stamp_ns = None
        # Dedup logic:
        # - If stamp matches previous AND arrival is within a very short window, treat as duplicate (dual QoS)
        # - Otherwise accept (covers cases where stamp is constant/zero in bags)
        if stamp_ns is not None and self._last_img_stamp_ns == stamp_ns and self._last_img_recv_ns is not None:
            if (now_ns - self._last_img_recv_ns) < int(2e6):  # 2ms window
                return
        self._last_img_stamp_ns = stamp_ns
        self._last_img_recv_ns = now_ns
        self.get_logger().debug(f'recv image: {msg.width}x{msg.height} ts={msg.header.stamp.sec}.{msg.header.stamp.nanosec}')
        self.last_img = msg
        if stamp_ns is not None:
            self.images_by_stamp[stamp_ns] = msg
            self._prune_buffer(self.images_by_stamp)
            # Try publish for this image timestamp
            self.try_publish(target_stamp_ns=stamp_ns)

    def cb_boxes(self, msg: BoundingBoxes):
        self.get_logger().debug(f'recv boxes: {len(msg.boxes)}')
        # cache boxes by timestamp for sync
        try:
            s = msg.header.stamp
            stamp_ns = int(getattr(s,'sec',0))*1_000_000_000 + int(getattr(s,'nanosec',0))
        except Exception:
            stamp_ns = None
        if stamp_ns is not None:
            self.boxes_by_stamp[stamp_ns] = list(msg.boxes)
            self._prune_buffer(self.boxes_by_stamp)
        # Try publish aligned to boxes timestamp
        if stamp_ns is not None:
            self.try_publish(target_stamp_ns=stamp_ns)

    def cb_tracks(self, msg: TrackArray):
        self.get_logger().debug(f'recv tracks: {len(msg.tracks)}')
        try:
            s = msg.header.stamp
            stamp_ns = int(getattr(s,'sec',0))*1_000_000_000 + int(getattr(s,'nanosec',0))
        except Exception:
            stamp_ns = None
        if stamp_ns is not None:
            self.tracks_by_stamp[stamp_ns] = list(msg.tracks)
            self._prune_buffer(self.tracks_by_stamp)
        # Try publish aligned to tracks timestamp
        if stamp_ns is not None:
            self.try_publish(target_stamp_ns=stamp_ns)

    def try_publish(self, target_stamp_ns: int = None):
        # Choose image by target stamp if provided; else use last_img
        img_msg = None
        tol_ns = int(self.sync_tol_ms * 1e6)
        if target_stamp_ns is not None and self.images_by_stamp:
            img_key = self._nearest_key(self.images_by_stamp, target_stamp_ns, tol_ns)
            img_msg = self.images_by_stamp.get(img_key)
        if img_msg is None:
            img_msg = self.last_img
        if img_msg is None:
            return
        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        h, w = img.shape[:2]
        # pick boxes/tracks matching current image timestamp (within tolerance)
        stamp_ns = None
        try:
            s = img_msg.header.stamp
            stamp_ns = int(getattr(s,'sec',0))*1_000_000_000 + int(getattr(s,'nanosec',0))
        except Exception:
            pass
        boxes_key = self._nearest_key(self.boxes_by_stamp, stamp_ns, tol_ns)
        tracks_key = self._nearest_key(self.tracks_by_stamp, stamp_ns, tol_ns)
        boxes = self.boxes_by_stamp.get(boxes_key, []) if boxes_key is not None else []
        tracks = self.tracks_by_stamp.get(tracks_key, []) if tracks_key is not None else []

        # If no boxes matched by timestamp, optionally hold the last known boxes within a time budget
        if (not boxes) and self.hold_last_boxes and self._last_boxes is not None and stamp_ns is not None and self._last_boxes_stamp_ns is not None:
            age_ns = abs(stamp_ns - self._last_boxes_stamp_ns)
            if age_ns <= int(self.hold_last_ms * 1e6):
                boxes = self._last_boxes

        # ==== draw boxes with class colors and threshold ====
        # Build filtered list (applies overlay_conf_min)
        vis_boxes = []
        for b in boxes:
            conf = float(getattr(b, 'probability', 0.0))
            if conf >= self.conf_min:
                vis_boxes.append(b)
        # Optional: if nothing passes threshold, re-use last visual boxes within hold window
        if not vis_boxes and self.hold_last_boxes and self.hold_on_empty and self._last_vis_boxes is not None and stamp_ns is not None and self._last_boxes_stamp_ns is not None:
            age_ns = abs(stamp_ns - self._last_boxes_stamp_ns)
            if age_ns <= int(self.hold_last_ms * 1e6):
                vis_boxes = self._last_vis_boxes
        for b in vis_boxes:
            name = getattr(b, 'class_id', '')
            conf = float(getattr(b, 'probability', 0.0))
            x1, y1, x2, y2 = int(b.xmin), int(b.ymin), int(b.xmax), int(b.ymax)
            x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
            y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
            color = _color_for(name)
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            label = f'{name} {conf:.2f}' if name else f'{conf:.2f}'
            cv2.putText(img, label, (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # ==== draw tracks (ID overlay) ====
        if tracks:
            for t in tracks:
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
        # Gate saving on detection count after overlay conf filter
        save_gate_ok = len(vis_boxes) >= int(self.save_min_boxes)
        if self.save_boxes_csv and self._csv_writer is not None and save_gate_ok:
            # Choose stamp: prefer boxes header (via matched key), else image header, else now
            if boxes_key is not None:
                sec = int(boxes_key // 1_000_000_000)
                nsec = int(boxes_key % 1_000_000_000)
            else:
                s = getattr(img_msg, 'header', None)
                sec = int(getattr(getattr(s,'stamp',s), 'sec', 0)) if s else 0
                nsec = int(getattr(getattr(s,'stamp',s), 'nanosec', 0)) if s else 0
                if sec == 0 and nsec == 0:
                    now = self.get_clock().now().to_msg()
                    sec, nsec = int(now.sec), int(now.nanosec)

            # Establish local zero based on first write moment
            if self._first_write_idx is None:
                self._first_write_idx = self._frame_idx
            local_idx = int(self._frame_idx - self._first_write_idx)
            for b in vis_boxes:
                self._csv_writer.writerow([
                    local_idx,  # normalized local frame (0-based)
                    local_idx,  # explicit frame_idx (0-based)
                    sec, nsec,
                    getattr(b,'class_id',''), float(getattr(b,'probability',0.0)),
                    int(b.xmin), int(b.ymin), int(b.xmax), int(b.ymax)
                ])
        if self.save_boxes_json and save_gate_ok:
            # JSON record includes local frame index and optional stamp
            if boxes_key is not None:
                sec = int(boxes_key // 1_000_000_000)
                nsec = int(boxes_key % 1_000_000_000)
            else:
                s = getattr(img_msg, 'header', None)
                sec = int(getattr(getattr(s,'stamp',s), 'sec', 0)) if s else 0
                nsec = int(getattr(getattr(s,'stamp',s), 'nanosec', 0)) if s else 0
                if sec == 0 and nsec == 0:
                    now = self.get_clock().now().to_msg()
                    sec, nsec = int(now.sec), int(now.nanosec)

            if self._first_write_idx is None:
                self._first_write_idx = self._frame_idx
            local_idx = int(self._frame_idx - self._first_write_idx)
            frame_record = []
            for b in vis_boxes:
                frame_record.append({
                    'frame': local_idx,
                    'sec': sec,
                    'nsec': nsec,
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
        if self._writer is not None and save_gate_ok:
            try:
                self._writer.write(img)
            except Exception as e:
                self.get_logger().warn(f'VideoWriter write failed: {e}')
        if self.save_images and save_gate_ok and (self._frame_idx % max(1, self.images_every_n) == 0):
            try:
                cv2.imwrite(os.path.join(self.images_dir, f'frame_{self._frame_idx:06d}.jpg'), img)
            except Exception as e:
                self.get_logger().warn(f'Image save failed: {e}')

        out = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        # 이미지 헤더 복사(타임싱크/리플레이 유용)
        out.header = img_msg.header
        self.pub_img.publish(out)
        self.pub_n += 1
        if self.pub_n % 30 == 0:
            self.get_logger().info(f'published #{self.pub_n} to {self.out_topic} - {img.shape[1]}x{img.shape[0]}, boxes={len(vis_boxes)}')

        # Update last boxes after publishing (use pre-filter boxes tied to actual matched key when available)
        try:
            if boxes_key is not None:
                self._last_boxes = self.boxes_by_stamp.get(boxes_key, None)
                self._last_boxes_stamp_ns = boxes_key
            elif boxes:
                # fall back to current image time if no discrete key but boxes used via hold
                self._last_boxes = list(boxes)
                self._last_boxes_stamp_ns = stamp_ns
            # store last visual (thresholded) boxes too for hold_on_empty
            if vis_boxes:
                self._last_vis_boxes = list(vis_boxes)
        except Exception:
            pass

    def _nearest_key(self, store: dict, stamp_ns: int, tol_ns: int):
        if stamp_ns is None or not store:
            return None
        # exact match preferred
        if stamp_ns in store:
            return stamp_ns
        # nearest within tolerance
        best_key = None
        best_dt = None
        for k in store.keys():
            dt = abs(k - stamp_ns)
            if best_dt is None or dt < best_dt:
                best_dt = dt; best_key = k
        if best_key is not None and best_dt <= tol_ns:
            return best_key
        return None

    def _prune_buffer(self, store: dict):
        # keep only most recent N entries
        if len(store) <= self.buffer_size:
            return
        keys = sorted(store.keys())
        for k in keys[:-self.buffer_size]:
            try:
                del store[k]
            except Exception:
                pass

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
                # Ensure directory exists before writing JSON
                json_dir = os.path.dirname(self.json_path) or '.'
                os.makedirs(json_dir, exist_ok=True)
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
