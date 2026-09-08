// 表情识别,订 /camera/image_raw,发 /emotion_result。

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "cv_bridge/cv_bridge.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "edge_inference_optimizer/latency_probe.hpp"
#include "edge_inference_optimizer/msg/emotion_result.hpp"
#include "edge_inference_optimizer/onnx_engine.hpp"
#include "edge_inference_optimizer/ros_trace.hpp"

namespace
{

// FER+ 八类顺序,少写一类后面全错位。
const std::vector<std::string> kFerPlusLabels = {
  "neutral", "happiness", "surprise", "sadness",
  "anger", "disgust", "fear", "contempt"};

/// 取最大的一张正脸。整图送进 FER+ 会给出很自信的噪声。
bool detect_largest_face(
  cv::CascadeClassifier & cascade, const cv::Mat & bgr, cv::Rect & out)
{
  if (cascade.empty()) { return false; }

  cv::Mat gray;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  cv::equalizeHist(gray, gray);   // Haar 特征对光照敏感,先做直方图均衡

  std::vector<cv::Rect> faces;
  cascade.detectMultiScale(gray, faces, 1.15, 4, 0, cv::Size(60, 60));
  if (faces.empty()) { return false; }

  out = *std::max_element(
    faces.begin(), faces.end(),
    [](const cv::Rect & a, const cv::Rect & b) { return a.area() < b.area(); });

  // 稍微放大:FER+ 的裁剪包含额头和下巴,Haar 的紧框会把两者切掉,
  // 系统性地影响靠眉毛和嘴部判别的类别。
  const int dx = static_cast<int>(out.width * 0.10);
  const int dy = static_cast<int>(out.height * 0.10);
  out.x -= dx; out.y -= dy; out.width += 2 * dx; out.height += 2 * dy;
  out &= cv::Rect(0, 0, bgr.cols, bgr.rows);   // 裁到图像范围内
  return out.width > 0 && out.height > 0;
}

void softmax_inplace(std::vector<float> & v)
{
  if (v.empty()) { return; }
  const float m = *std::max_element(v.begin(), v.end());
  float sum = 0.0f;
  for (auto & x : v) { x = std::exp(x - m); sum += x; }   // 减最大值防溢出
  if (sum > 0.0f) { for (auto & x : v) { x /= sum; } }
}

}  // namespace

class EmotionDetectorNode : public rclcpp::Node
{
public:
  EmotionDetectorNode()
  : Node("emotion_detector")
  {
    const auto model = declare_parameter<std::string>("model_path", "");
    const auto backend = declare_parameter<std::string>("backend", "cpu");
    const auto threads = declare_parameter<int>("threads", 1);
    input_topic_ = declare_parameter<std::string>("input_topic", "/camera/image_raw");
    output_topic_ = declare_parameter<std::string>("output_topic", "/emotion_result");

    // 路径为空则关闭检测,退化为整帧分类——只有当输入本身就是人脸裁剪时才有意义。
    const auto cascade_path = declare_parameter<std::string>("face_cascade", "");
    if (!cascade_path.empty()) {
      if (!cascade_.load(cascade_path)) {
        RCLCPP_ERROR(
          get_logger(), "failed to load cascade '%s'; falling back to whole-frame input",
          cascade_path.c_str());
      } else {
        RCLCPP_INFO(get_logger(), "face detection enabled: %s", cascade_path.c_str());
      }
    } else {
      RCLCPP_WARN(
        get_logger(),
        "未启用人脸检测,整帧将被压缩到 %dx%d。FER+ 需要人脸裁剪作为输入,"
        "除非输入本身就是人脸,否则预测结果没有意义。", 64, 64);
    }

    if (model.empty()) {
      RCLCPP_FATAL(get_logger(), "parameter 'model_path' is required");
      throw std::runtime_error("model_path not set");
    }

    eio::OnnxEngine::Options opts;
    opts.model_path = model;
    opts.backend = (backend == "cuda") ? eio::Backend::kCuda : eio::Backend::kCpu;
    opts.intra_op_threads = threads;
    opts.inter_op_threads = threads;
    engine_ = std::make_unique<eio::OnnxEngine>(opts);

    RCLCPP_INFO(get_logger(), "%s", engine_->describe().c_str());
    if (opts.backend == eio::Backend::kCuda && engine_->backend() != eio::Backend::kCuda) {
      RCLCPP_WARN(
        get_logger(), "CUDA requested but running on CPU: %s",
        engine_->cuda_fallback_reason().c_str());
    }

    // 输入尺寸从模型读,不写死 64x64:换模型时预处理应自动跟着变,
    // 而不是产出静默错误的裁剪。
    const auto & in = engine_->inputs().at(0);
    if (in.shape.size() != 4) {
      throw std::runtime_error("expected a 4-D NCHW input, got rank " +
              std::to_string(in.shape.size()));
    }
    channels_ = static_cast<int>(in.shape[1] < 0 ? 1 : in.shape[1]);
    height_ = static_cast<int>(in.shape[2] < 0 ? 64 : in.shape[2]);
    width_ = static_cast<int>(in.shape[3] < 0 ? 64 : in.shape[3]);

    pub_ = create_publisher<edge_inference_optimizer::msg::EmotionResult>(output_topic_, 10);
    sub_ = create_subscription<sensor_msgs::msg::Image>(
      input_topic_, rclcpp::SensorDataQoS(),
      std::bind(&EmotionDetectorNode::on_image, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(), "listening on %s, publishing %s, input %dx%dx%d",
      input_topic_.c_str(), output_topic_.c_str(), channels_, height_, width_);
  }

private:
  void on_image(const sensor_msgs::msg::Image::ConstSharedPtr & img)
  {
    const rclcpp::Time now = this->now();
    eio::LatencyTrace trace = eio::LatencyTrace::start(++seq_);

    // 从 camera 节点发布到本节点被唤醒之间,消耗在 DDS 上的时间
    const double transport = eio::transport_ms(rclcpp::Time(img->header.stamp), now);

    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::BGR8);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge failed: %s", e.what());
      return;
    }

    cv::Rect face;
    bool have_face = false;
    double detect_ms = 0.0;
    if (!cascade_.empty()) {
      eio::ScopedStage stage(trace, "face_detect");
      have_face = detect_largest_face(cascade_, cv_ptr->image, face);
      detect_ms = stage.commit();

      if (!have_face) {
        // 没有人脸就没有表情。把基于背景推出来的标签发出去,
        // 等于往舵机指令流和延迟统计里塞一个编造的读数。
        ++frames_without_face_;
        if ((frames_without_face_ % 60) == 1) {
          RCLCPP_WARN(
            get_logger(), "no face detected (%lu frames so far); skipping classification",
            static_cast<unsigned long>(frames_without_face_));
        }
        return;
      }
    }

    std::vector<float> tensor;
    double preprocess_ms = 0.0;
    {
      eio::ScopedStage stage(trace, "preprocess");
      try {
        const cv::Mat roi = have_face ? cv_ptr->image(face) : cv_ptr->image;
        tensor = preprocess(roi);
      } catch (const std::exception & e) {
        RCLCPP_ERROR(get_logger(), "preprocess failed: %s", e.what());
        return;
      }
      preprocess_ms = stage.commit();
    }

    std::vector<float> scores;
    double infer_ms = 0.0;
    {
      eio::ScopedStage stage(trace, "onnx_infer");
      try {
        scores = engine_->run(
          tensor,
          {1, static_cast<int64_t>(channels_), height_, width_});
      } catch (const std::exception & e) {
        RCLCPP_ERROR(get_logger(), "inference failed: %s", e.what());
        return;
      }
      infer_ms = stage.commit();
    }

    // 模型输出的是原始 logits,归一化之后置信度才有意义
    softmax_inplace(scores);

    const auto best = std::max_element(scores.begin(), scores.end());
    const auto idx = static_cast<size_t>(std::distance(scores.begin(), best));

    edge_inference_optimizer::msg::EmotionResult out;
    out.header.stamp = now;
    out.header.frame_id = img->header.frame_id;
    out.emotion = idx < kFerPlusLabels.size() ? kFerPlusLabels[idx] : "unknown";
    out.confidence = *best;
    out.scores = scores;
    out.labels = kFerPlusLabels;
    out.preprocess_ms = static_cast<float>(preprocess_ms);
    out.inference_ms = static_cast<float>(infer_ms);

    out.trace = eio::to_msg(trace, rclcpp::Time(img->header.stamp));
    // 把传输段插到本地各段之前,让瀑布图从「这一帧产生的时刻」开始,
    // 而不是从「我们看到它的时刻」开始。
    eio::append_stage(out.trace, "dds_transport_camera", 0.0, transport, /*is_compute=*/false);
    std::rotate(out.trace.stages.begin(), out.trace.stages.end() - 1, out.trace.stages.end());
    for (size_t i = 1; i < out.trace.stages.size(); ++i) {
      out.trace.stages[i].start_ms += transport;
    }
    out.trace.total_ms = transport + detect_ms + preprocess_ms + infer_ms;

    pub_->publish(out);

    if ((seq_ % 30) == 1) {
      RCLCPP_INFO(
        get_logger(),
        "%-10s conf=%.3f | detect %.2f | pre %.2f | infer %.2f ms | face %s",
        out.emotion.c_str(), out.confidence, detect_ms, preprocess_ms, infer_ms,
        have_face ? (std::to_string(face.width) + "x" + std::to_string(face.height)).c_str()
                  : "(whole frame)");
    }
  }

  std::vector<float> preprocess(const cv::Mat & bgr) const
  {
    cv::Mat gray;
    if (channels_ == 1) {
      cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    } else {
      gray = bgr;
    }

    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(width_, height_), 0, 0, cv::INTER_LINEAR);

    cv::Mat f32;
    // FER+ 要 0-255 灰度,不要除 255。
    resized.convertTo(f32, CV_32F);

    std::vector<float> out(static_cast<size_t>(channels_ * height_ * width_));
    if (channels_ == 1) {
      std::memcpy(out.data(), f32.ptr<float>(), out.size() * sizeof(float));
    } else {
      // NHWC(OpenCV)转 NCHW(ONNX)
      std::vector<cv::Mat> planes;
      cv::split(f32, planes);
      size_t off = 0;
      for (int c = 0; c < channels_ && c < static_cast<int>(planes.size()); ++c) {
        const size_t n = static_cast<size_t>(height_ * width_);
        std::memcpy(out.data() + off, planes[c].ptr<float>(), n * sizeof(float));
        off += n;
      }
    }
    return out;
  }

  std::unique_ptr<eio::OnnxEngine> engine_;
  cv::CascadeClassifier cascade_;
  rclcpp::Publisher<edge_inference_optimizer::msg::EmotionResult>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  std::string input_topic_, output_topic_;
  int channels_ = 1, height_ = 64, width_ = 64;
  uint64_t seq_ = 0;
  uint64_t frames_without_face_ = 0;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<EmotionDetectorNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("emotion_detector"), "fatal: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
