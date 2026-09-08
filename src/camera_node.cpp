// 发图像。图片目录、视频、摄像头都行。

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "cv_bridge/cv_bridge.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

class CameraNode : public rclcpp::Node
{
public:
  CameraNode()
  : Node("camera_node")
  {
    const auto fps = declare_parameter<double>("fps", 30.0);
    image_dir_ = declare_parameter<std::string>("image_dir", "");
    video_path_ = declare_parameter<std::string>("video_path", "");
    device_id_ = declare_parameter<int>("device_id", -1);
    width_ = declare_parameter<int>("width", 640);
    height_ = declare_parameter<int>("height", 480);
    const auto topic = declare_parameter<std::string>("output_topic", "/camera/image_raw");

    pub_ = create_publisher<sensor_msgs::msg::Image>(topic, rclcpp::SensorDataQoS());

    if (!image_dir_.empty()) {
      load_dir();
    } else if (!video_path_.empty()) {
      cap_.open(video_path_);
      if (!cap_.isOpened()) {
        RCLCPP_ERROR(get_logger(), "cannot open video %s; using synthetic frames",
          video_path_.c_str());
      }
    } else if (device_id_ >= 0) {
      cap_.open(device_id_);
      if (!cap_.isOpened()) {
        RCLCPP_ERROR(get_logger(), "cannot open device %d; using synthetic frames", device_id_);
      }
    }

    const auto period = std::chrono::duration<double>(fps > 0.0 ? 1.0 / fps : 1.0 / 30.0);
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&CameraNode::tick, this));

    RCLCPP_INFO(get_logger(), "publishing %s at %.1f fps (%s)", topic.c_str(), fps,
      source_name().c_str());
  }

private:
  std::string source_name() const
  {
    if (!files_.empty()) { return "image_dir:" + image_dir_; }
    if (cap_.isOpened()) { return video_path_.empty() ? "device" : "video:" + video_path_; }
    return "synthetic";
  }

  void load_dir()
  {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::is_directory(image_dir_, ec)) {
      RCLCPP_ERROR(get_logger(), "image_dir %s is not a directory", image_dir_.c_str());
      return;
    }
    for (const auto & e : fs::directory_iterator(image_dir_, ec)) {
      const auto ext = e.path().extension().string();
      if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
        files_.push_back(e.path().string());
      }
    }
    std::sort(files_.begin(), files_.end());
    RCLCPP_INFO(get_logger(), "loaded %zu images from %s", files_.size(), image_dir_.c_str());
  }

  cv::Mat synthetic()
  {
    cv::Mat img(height_, width_, CV_8UC3, cv::Scalar(40, 40, 45));
    const double t = static_cast<double>(seq_) * 0.08;
    const int cx = width_ / 2 + static_cast<int>(40 * std::sin(t));
    const int cy = height_ / 2 + static_cast<int>(20 * std::cos(t * 0.7));
    const int r = std::min(width_, height_) / 4;
    cv::circle(img, {cx, cy}, r, cv::Scalar(180, 175, 170), -1);
    cv::circle(img, {cx - r / 2, cy - r / 3}, r / 8, cv::Scalar(20, 20, 20), -1);
    cv::circle(img, {cx + r / 2, cy - r / 3}, r / 8, cv::Scalar(20, 20, 20), -1);
    cv::ellipse(img, {cx, cy + r / 3}, {r / 2, r / 4},
      0, 0, 180 + 60 * std::sin(t), cv::Scalar(20, 20, 20), 3);
    return img;
  }

  void tick()
  {
    cv::Mat frame;
    if (!files_.empty()) {
      frame = cv::imread(files_[seq_ % files_.size()], cv::IMREAD_COLOR);
      if (frame.empty()) { frame = synthetic(); }
    } else if (cap_.isOpened()) {
      if (!cap_.read(frame) || frame.empty()) {
        cap_.set(cv::CAP_PROP_POS_FRAMES, 0);   // 循环播放
        if (!cap_.read(frame) || frame.empty()) { frame = synthetic(); }
      }
    } else {
      frame = synthetic();
    }

    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
    // 时间戳打在采集时刻而非发布时刻:下游都以此为基准算延迟,
    // 打晚了会把本节点自身的耗时从端到端总时长里悄悄减掉。
    msg->header.stamp = this->now();
    msg->header.frame_id = "camera";
    pub_->publish(*msg);
    ++seq_;
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  cv::VideoCapture cap_;
  std::vector<std::string> files_;
  std::string image_dir_, video_path_;
  int device_id_ = -1, width_ = 640, height_ = 480;
  uint64_t seq_ = 0;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraNode>());
  rclcpp::shutdown();
  return 0;
}
