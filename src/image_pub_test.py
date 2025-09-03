import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import glob
import os

class ImagePub(Node):
    def __init__(self):
        super().__init__('image_pub_test')
        self.publisher_ = self.create_publisher(Image, '/image_raw', 10)
        # 퍼블리시 주기 파라미터화
        self.frequency = float(self.declare_parameter('frequency', 20.0).get_parameter_value().double_value)
        period = 0.05
        try:
            if self.frequency > 0:
                period = 1.0 / float(self.frequency)
        except Exception:
            pass
        self.timer = self.create_timer(period, self.timer_callback)
        self.bridge = CvBridge()

        # ✅ 데이터셋 선택 파라미터
        # split: train/val/test (Argoverse 기본 경로),
        # root: 커스텀 데이터셋 루트(주어지면 split 대신 사용),
        # patterns: 쉼표로 구분된 glob 패턴 목록(예: "**/*.jpg,**/*.png"),
        # list_file: 이미지 절대경로 목록 파일(줄바꿈 구분)
        self.split = self.declare_parameter('split', 'train').get_parameter_value().string_value
        self.root  = self.declare_parameter('root', '').get_parameter_value().string_value
        self.patterns = self.declare_parameter('patterns', '**/*.jpg,**/*.jpeg,**/*.png').get_parameter_value().string_value
        self.list_file = self.declare_parameter('list_file', '').get_parameter_value().string_value
        self.recursive = self.declare_parameter('recursive', True).get_parameter_value().bool_value
        self.shuffle = self.declare_parameter('shuffle', False).get_parameter_value().bool_value
        self.seed = int(self.declare_parameter('seed', 42).get_parameter_value().integer_value)

        # 기본은 Argoverse. 필요 시 bdd도 옵션으로 포함 가능
        self.include_bdd = self.declare_parameter('include_bdd', False).get_parameter_value().bool_value
        # 한 바퀴 후 종료 여부
        self.loop = self.declare_parameter('loop', False).get_parameter_value().bool_value

        # 경로 수집 로직
        img_paths = []
        if self.list_file and os.path.isfile(self.list_file):
            try:
                with open(self.list_file, 'r') as f:
                    img_paths = [ln.strip() for ln in f.readlines() if ln.strip()]
                img_paths = [p for p in img_paths if os.path.isfile(p)]
            except Exception as e:
                self.get_logger().warn(f"[image_pub_test] list_file 읽기 실패: {e}")
        else:
            root = self.root.strip()
            if not root:
                # 'valid' 표기도 허용 -> 'val'로 매핑
                split_dir = self.split
                if split_dir.lower() == 'valid':
                    split_dir = 'val'
                root = f"/workspace/dataset/Argoverse-HD-Full/Argoverse-1.1/tracking/{split_dir}"
            pats = [p.strip() for p in self.patterns.split(',') if p.strip()]
            for pat in pats:
                # 패턴이 '**'를 포함하면 recursive True가 자동 필요
                rec = self.recursive or ('**' in pat)
                img_paths += glob.glob(os.path.join(root, pat), recursive=rec)

            if self.include_bdd:
                bdd_root = "/workspace/dataset/bdd100k"
                for pat in pats:
                    rec = self.recursive or ('**' in pat)
                    img_paths += glob.glob(os.path.join(bdd_root, pat), recursive=rec)

        # 정렬/셔플
        try:
            img_paths = sorted(set(img_paths))
        except Exception:
            pass
        if self.shuffle:
            try:
                import random
                random.seed(self.seed)
                random.shuffle(img_paths)
            except Exception:
                pass
        self.img_paths = img_paths

        self.idx = 0
        self.log_every = int(self.declare_parameter('log_every', 30).get_parameter_value().integer_value)
        base_info = f"split={self.split}" if not self.root else f"root={self.root}"
        self.get_logger().info(f"[image_pub_test] {base_info}, files={len(self.img_paths)}")
        if self.img_paths:
            self.get_logger().info(f"[image_pub_test] first={self.img_paths[0]}")

    def timer_callback(self):
        if not self.img_paths:
            self.get_logger().warn("[image_pub_test] 이미지가 없습니다. split 경로나 확장자를 확인하세요.")
            return
        # 이미지 순차적으로 불러오기
        path = self.img_paths[self.idx]
        img = cv2.imread(path)
        if img is None:
            self.get_logger().info(f"이미지 로딩 실패: {path}")
            self.idx = (self.idx + 1) % len(self.img_paths)
            return
        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        # Ensure valid timestamp/frame_id for proper synchronization downstream
        try:
            msg.header.stamp = self.get_clock().now().to_msg()
        except Exception:
            pass
        if not getattr(msg.header, 'frame_id', None):
            msg.header.frame_id = 'camera'
        self.publisher_.publish(msg)
        if self.log_every > 0 and (self.idx % self.log_every == 0):
            self.get_logger().info(f'퍼블리시: {path}')
        # 다음 인덱스 계산 및 한 바퀴 종료 처리
        next_idx = (self.idx + 1) % len(self.img_paths)
        finished_round = (next_idx == 0)
        self.idx = next_idx
        if finished_round and not self.loop:
            try:
                self.get_logger().info('[image_pub_test] dataset 한 바퀴 완료. 퍼블리셔 종료합니다.')
                self.timer.cancel()
            except Exception:
                pass
            # rclpy.shutdown()을 호출하면 main의 spin에서 빠져나옵니다.
            try:
                rclpy.shutdown()
            except Exception:
                pass

def main(args=None):
    rclpy.init(args=args)
    node = ImagePub()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
