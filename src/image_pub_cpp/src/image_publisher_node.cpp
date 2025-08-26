#include <chrono>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

using namespace std::chrono_literals;

class ImagePublisher : public rclcpp::Node
{
public:
  ImagePublisher()
  : Node("image_publisher_node"), count_(0)
  {
    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("image_raw", 10);
    timer_ = this->create_wall_timer(
      500ms, std::bind(&ImagePublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {
    cv::Mat image(480, 640, CV_8UC3, cv::Scalar(200, 100, 100));  // 단색 이미지

    std_msgs::msg::Header header;
    header.stamp = this->get_clock()->now();
    header.frame_id = "camera_frame";

    auto msg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
    publisher_->publish(*msg);
    RCLCPP_INFO(this->get_logger(), "Published image #%zu", count_++);
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  size_t count_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImagePublisher>());
  rclcpp::shutdown();
  return 0;
}