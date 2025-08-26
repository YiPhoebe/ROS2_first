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
        self.timer = self.create_timer(0.05, self.timer_callback) # <- 시간 조절 안정성 확보 위해 1.0초로 했는데 49시간 걸려서.. 빠른 테스트 위해 0.05초로 변경함
        self.bridge = CvBridge()

        # ✅ split 파라미터(train/val/test)로 경로 전환
        self.split = self.declare_parameter('split', 'train').get_parameter_value().string_value
        root = f"/workspace/dataset/Argoverse-HD-Full/Argoverse-1.1/tracking/{self.split}"
        # 기본은 Argoverse만 사용. 필요 시 bdd도 옵션으로 포함 가능
        self.include_bdd = self.declare_parameter('include_bdd', False).get_parameter_value().bool_value

        self.img_paths = sorted(
            glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)
            + glob.glob(os.path.join(root, "**", "*.jpeg"), recursive=True)
            + glob.glob(os.path.join(root, "**", "*.png"), recursive=True)
        )

        if self.include_bdd:
            bdd_root = "/workspace/dataset/bdd100k"
            self.img_paths += sorted(glob.glob(os.path.join(bdd_root, "**", "*.jpg"), recursive=True))

        self.idx = 0
        self.get_logger().info(f"[image_pub_test] split={self.split}, files={len(self.img_paths)}")
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
        self.publisher_.publish(msg)
        self.get_logger().info(f'퍼블리시: {path}')
        self.idx = (self.idx + 1) % len(self.img_paths)

def main(args=None):
    rclpy.init(args=args)
    node = ImagePub()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()