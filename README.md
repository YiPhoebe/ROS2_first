
# ROS2 기반 비디오 인식 파이프라인 프로젝트

## 프로젝트 개요
이 프로젝트는 ROS2 (Iron) 환경에서 비디오 스트림을 받아 YOLO 기반 객체 검출 및 멀티 오브젝트 트래킹(MOT), 결과 Overlay, 저장까지의 파이프라인을 구현한 예제입니다. 실시간 인식 시스템의 구조와 동작 방식을 학습하고, 다양한 환경에서 손쉽게 개발 및 테스트할 수 있도록 설계되었습니다.

## 개발 및 실행 환경
- **Docker**: 개발 환경의 일관성을 위해 Docker 컨테이너(ros:iron-ros-base 이미지)에서 실행
- **VSCode**: 소스 코드 편집 및 원격 개발(Dev Containers)
- **Mac M3**: Apple Silicon(M3) 기반 Mac에서 개발 및 테스트
- **ROS2 Iron**: 최신 ROS2 Iron 배포판 기반

## 파이프라인 구조
```
[Publisher (Video/Image)] 
      ↓
[YOLO Inference Node] 
      ↓
[Overlay Node] 
      ↓
[MOT (Multi-Object Tracking) Node] 
      ↓
[Result Saver Node]
```
각 노드는 ROS2의 Publisher/Subscriber 구조로 메시지를 주고받으며, 각 단계별로 결과를 시각화하거나 저장할 수 있습니다.

## 주요 기능
- **Publisher**: 비디오 파일 또는 카메라 스트림을 ROS2 토픽으로 퍼블리시
- **YOLO 추론**: YOLO 모델을 이용해 입력 프레임에서 객체 검출
- **Overlay**: 검출 결과(바운딩 박스, 클래스 등)를 영상에 Overlay
- **MOT (Multi-Object Tracking)**: 객체 추적 알고리즘으로 각 객체의 ID 및 이동 경로 추적
- **저장**: 최종 결과 영상을 파일로 저장

## 실행 방법
1. **Docker 컨테이너 실행**
   ```bash
   docker compose up -d
   docker exec -it ros2_dev bash
   ```
2. **의존성 설치 및 빌드**
   ```bash
   cd ~/ros2_ws
   rosdep install --from-paths src --ignore-src -r -y
   colcon build
   source install/setup.bash
   ```
3. **노드 실행**
   - 퍼블리셔 실행:
     ```bash
     ros2 run video_publisher video_publisher_node --ros-args -p video_path:=/path/to/video.mp4
     ```
   - YOLO 노드 실행:
     ```bash
     ros2 run yolo_inference yolo_inference_node
     ```
   - Overlay, MOT, 저장 노드 등도 각각 실행 (예시 생략)

## 결과 예시
아래는 파이프라인을 통해 처리된 결과 영상의 예시입니다.
![result_example](docs/result_example.png)
- 검출된 객체에 바운딩 박스와 클래스 라벨, 추적 ID가 Overlay되어 표시됩니다.
- 결과 영상은 지정된 디렉토리에 mp4 파일로 저장됩니다.

## 향후 계획
- YOLO 모델 경량화 및 속도 최적화
- 다양한 Tracking 알고리즘 지원(DeepSORT 등)
- Web UI를 통한 실시간 결과 모니터링 기능 추가
- 다양한 센서(예: LiDAR, Depth Camera)와의 통합
- 자동화된 테스트 및 배포 파이프라인 구축

---
문의 및 피드백은 Issues 또는 Pull Request로 남겨주세요.
