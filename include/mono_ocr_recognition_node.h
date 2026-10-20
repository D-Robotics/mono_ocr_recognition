// Copyright (c) 2026, D-Robotics.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MONO_OCR_RECOGNITION_NODE_H_
#define MONO_OCR_RECOGNITION_NODE_H_

#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

#include "dnn_node/dnn_node.h"
#include "dnn_node/util/image_proc.h"
#include "opencv2/core/mat.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "ai_msgs/msg/perception_targets.hpp"
#include "ai_msgs/msg/perf.hpp"
#include "dnn_node/dnn_node_data.h"

#ifdef SHARED_MEM_ENABLED
#include "hbm_img_msgs/msg/hbm_msg1080_p.hpp"
#include "hbm_img_msgs/msg/hbm_msg2160_p.hpp"
#endif

#include "include/post_process/mono_ocr_recognition_output_parser.h"

using rclcpp::NodeOptions;
using hobot::dnn_node::DnnNode;
using hobot::dnn_node::DnnNodeOutput;
using hobot::dnn_node::DNNInput;
using hobot::dnn_node::NV12PyramidInput;

struct MonoOcrRecognitionOutput : public DnnNodeOutput {
  int model_h = 0, model_w = 0;
  // 该批 ROI 对应的 det 消息（PostProcess 里回填文字后发布）
  ai_msgs::msg::PerceptionTargets::ConstSharedPtr det_msg;
  // 该批 ROI 的文本框数量（= output_tensors 里 batch 的数量）
  int roi_count = 0;
};

class MonoOcrRecognitionNode : public DnnNode {
 public:
  MonoOcrRecognitionNode(const std::string &node_name,
          const NodeOptions &options = NodeOptions());
  // component 容器加载入口
  explicit MonoOcrRecognitionNode(const rclcpp::NodeOptions &options)
      : MonoOcrRecognitionNode("mono_ocr_recognition", options) {}
  ~MonoOcrRecognitionNode() override;

 protected:
  int SetNodePara() override;
  int PostProcess(const std::shared_ptr<DnnNodeOutput> &outputs) override;

 private:
  int LoadConfig();
  int FeedFromLocal();  // feed_type=0：本地图整图识别（独立验证用）

  // 订阅 det 节点的文本框结果
  void AiMsgProcess(
      const ai_msgs::msg::PerceptionTargets::ConstSharedPtr msg);
  rclcpp::Subscription<ai_msgs::msg::PerceptionTargets>::SharedPtr
      ai_msg_subscription_ = nullptr;

  // 订阅图像（NV12），缓存最近一帧供 det 框裁剪
  void RosImgProcess(const sensor_msgs::msg::Image::ConstSharedPtr msg);
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
      ros_img_subscription_ = nullptr;

#ifdef SHARED_MEM_ENABLED
  void SharedMemImgProcess(
      const hbm_img_msgs::msg::HbmMsg2160P::ConstSharedPtr msg);
  // 公共处理：缓存整帧 NV12 pyramid（resizer 模型输入）
  void ProcessSharedMemNv12(const uint8_t *data, uint32_t height,
                            uint32_t width, const std::string &encoding);
  rclcpp::Subscription<hbm_img_msgs::msg::HbmMsg2160P>::SharedPtr
      sharedmem_img_subscription_4k_ = nullptr;
#endif

  int is_shared_mem_sub_ = 0;
  std::string sharedmem_img_topic_name_ = "/hbmem_img";
  // 4K 源下 web 显示的帧：fb=半分辨率副本(默认)，usb=全分辨率帧
  // web 链路显示全分辨率帧，发布坐标不再缩到降采样副本空间
  bool display_full_resolution_ = true;

  // 最近一帧 NV12 pyramid 缓存（resizer 模型输入需要整图 pyramid，非裁剪 BGR）
  std::shared_ptr<NV12PyramidInput> cache_pyramid_;
  std::mutex cache_img_mtx_;
  bool has_cache_img_ = false;
  int cache_img_h_ = 0;
  int cache_img_w_ = 0;
  // 4K 模式：rec 用半分辨率 pyramid（绕开 resizer ROI 宽度 ≤3.5x320 限制），
  // 记录 det 全帧尺寸用于 ROI 坐标换算
  int full_frame_h_ = 0;
  int full_frame_w_ = 0;
  // 全分辨率 NV12 帧缓存（软件路径逐框裁剪用，等比缩放+补黑复刻 PaddleOCR 预处理）
  cv::Mat cache_full_;

  std::string config_file_ = "config/mono_ocr_recognition_workconfig.json";
  std::string model_file_name_ = "";
  std::string model_name_ = "";
  std::string char_dict_file_ = "";
  int task_num_ = 4;
  int is_sync_mode_ = 0;
  int feed_type_ = 1;  // 0=本地图整图识别（独立验证用），1=订阅 det 节点的文本框结果

  int model_input_width_ = 320;
  int model_input_height_ = 48;

  std::string ai_msg_sub_topic_name_ = "/hobot_ocr_detection";
  std::string ai_msg_pub_topic_name_ = "/hobot_ocr_recognition";
  std::string ros_img_sub_topic_name_ = "/image";
  rclcpp::Publisher<ai_msgs::msg::PerceptionTargets>::SharedPtr
      msg_publisher_ = nullptr;
};

#endif  // MONO_OCR_RECOGNITION_NODE_H_
