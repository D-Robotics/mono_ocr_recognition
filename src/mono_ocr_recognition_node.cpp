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

#include "include/mono_ocr_recognition_node.h"
#include "hobot_cv/hobotcv_imgproc.h"

#include <unistd.h>
#include <chrono>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "ament_index_cpp/get_package_prefix.hpp"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

#include "dnn_node/dnn_node.h"
#include "dnn_node/util/image_proc.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgproc.hpp"
#include "rclcpp/rclcpp.hpp"

MonoOcrRecognitionNode::MonoOcrRecognitionNode(
    const std::string &node_name, const NodeOptions &options)
    : DnnNode(node_name, options) {
  this->declare_parameter<std::string>("config_file", config_file_);
  this->declare_parameter<int>("is_sync_mode", is_sync_mode_);
  this->declare_parameter<int>("feed_type", feed_type_);
  this->declare_parameter<std::string>("ai_msg_sub_topic_name",
                                       ai_msg_sub_topic_name_);
  this->declare_parameter<std::string>("ai_msg_pub_topic_name",
                                       ai_msg_pub_topic_name_);
  this->declare_parameter<std::string>("ros_img_sub_topic_name",
                                       ros_img_sub_topic_name_);
  this->declare_parameter<int>("is_shared_mem_sub", is_shared_mem_sub_);
  this->declare_parameter<bool>("display_full_resolution", display_full_resolution_);
  this->declare_parameter<int>("model_input_width", model_input_width_);
  this->declare_parameter<int>("model_input_height", model_input_height_);
  this->declare_parameter<std::string>("sharedmem_img_topic_name",
                                       sharedmem_img_topic_name_);
  this->get_parameter<std::string>("config_file", config_file_);
  this->get_parameter<int>("is_sync_mode", is_sync_mode_);
  this->get_parameter<int>("feed_type", feed_type_);
  this->get_parameter<std::string>("ai_msg_sub_topic_name",
                                   ai_msg_sub_topic_name_);
  this->get_parameter<std::string>("ai_msg_pub_topic_name",
                                   ai_msg_pub_topic_name_);
  this->get_parameter<std::string>("ros_img_sub_topic_name",
                                   ros_img_sub_topic_name_);
  this->get_parameter<int>("is_shared_mem_sub", is_shared_mem_sub_);
  this->get_parameter<bool>("display_full_resolution", display_full_resolution_);
  this->get_parameter<int>("model_input_width", model_input_width_);
  this->get_parameter<int>("model_input_height", model_input_height_);
  this->get_parameter<std::string>("sharedmem_img_topic_name",
                                   sharedmem_img_topic_name_);

  if (LoadConfig() != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("mono_ocr_recognition"),
                 "Load config fail!");
    rclcpp::shutdown();
    return;
  }
  // 加载字符字典
  if (mono_ocr_recognition::LoadCharacterDict(char_dict_file_) != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("mono_ocr_recognition"),
                 "Load char dict fail!");
    rclcpp::shutdown();
    return;
  }

  if (Init() != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("mono_ocr_recognition"), "Init failed!");
    rclcpp::shutdown();
    return;
  }
  // 注意：NV12 双输入模型的 GetModelInputSize 会解析错误，直接用 config 写死
  // 的 48x320。
  RCLCPP_INFO(rclcpp::get_logger("mono_ocr_recognition"),
              "Model input: %d x %d (from config)", model_input_width_,
              model_input_height_);

  msg_publisher_ = this->create_publisher<ai_msgs::msg::PerceptionTargets>(
      ai_msg_pub_topic_name_, 10);

  if (feed_type_ == 0) {
    FeedFromLocal();
  } else {
    // 订阅 det 节点的文本框结果
    ai_msg_subscription_ =
        this->create_subscription<ai_msgs::msg::PerceptionTargets>(
            ai_msg_sub_topic_name_, 10,
            std::bind(&MonoOcrRecognitionNode::AiMsgProcess, this,
                      std::placeholders::_1));
    // 订阅图像
    if (is_shared_mem_sub_) {
#ifdef SHARED_MEM_ENABLED
      sharedmem_img_subscription_4k_ =
          this->create_subscription<hbm_img_msgs::msg::HbmMsg2160P>(
              sharedmem_img_topic_name_, rclcpp::SensorDataQoS().keep_last(1),
              std::bind(&MonoOcrRecognitionNode::SharedMemImgProcess, this,
                        std::placeholders::_1));
      RCLCPP_WARN(rclcpp::get_logger("mono_ocr_recognition"),
                  "Subscribe 4K shared mem img: %s (HbmMsg2160P)",
                  sharedmem_img_topic_name_.c_str());
#else
      RCLCPP_ERROR(rclcpp::get_logger("mono_ocr_recognition"),
                   "Unsupport shared mem");
#endif
    } else {
      ros_img_subscription_ =
          this->create_subscription<sensor_msgs::msg::Image>(
              ros_img_sub_topic_name_, 10,
              std::bind(&MonoOcrRecognitionNode::RosImgProcess, this,
                        std::placeholders::_1));
      RCLCPP_INFO(rclcpp::get_logger("mono_ocr_recognition"),
                  "Subscribe img: %s, det: %s", ros_img_sub_topic_name_.c_str(),
                  ai_msg_sub_topic_name_.c_str());
    }
  }
}

MonoOcrRecognitionNode::~MonoOcrRecognitionNode() {}

int MonoOcrRecognitionNode::FeedFromLocal() {
  // resizer 模型需要 ROI，feed_type=0 本地图模式不适用，仅保留接口
  RCLCPP_WARN(rclcpp::get_logger("mono_ocr_recognition"),
              "feed_type=0 not supported for resizer model, use feed_type=1");
  return -1;
}

int MonoOcrRecognitionNode::LoadConfig() {
  if (config_file_.empty()) return -1;
  std::ifstream ifs(config_file_.c_str());
  if (!ifs) {
    RCLCPP_ERROR(rclcpp::get_logger("mono_ocr_recognition"),
                 "Read config file [%s] fail!", config_file_.c_str());
    return -1;
  }
  rapidjson::IStreamWrapper isw(ifs);
  rapidjson::Document doc;
  doc.ParseStream(isw);
  if (doc.HasParseError()) return -1;
  if (doc.HasMember("model_file"))
    model_file_name_ = doc["model_file"].GetString();
  if (doc.HasMember("model_name"))
    model_name_ = doc["model_name"].GetString();
  if (doc.HasMember("char_dict_file"))
    char_dict_file_ = doc["char_dict_file"].GetString();
  if (doc.HasMember("task_num")) task_num_ = doc["task_num"].GetInt();
  // 模型输入宽高（640x48 宽行档 / 320x48 基础档），软件路径与 CTC 时间步都依赖
  if (doc.HasMember("model_input_width"))
    model_input_width_ = doc["model_input_width"].GetInt();
  if (doc.HasMember("model_input_height"))
    model_input_height_ = doc["model_input_height"].GetInt();
  return mono_ocr_recognition::LoadConfig(doc);
}

int MonoOcrRecognitionNode::SetNodePara() {
  if (!dnn_node_para_ptr_) return -1;
  dnn_node_para_ptr_->model_file = model_file_name_;
  dnn_node_para_ptr_->model_name = model_name_;
  // resizer 模型：输入 = 整图 NV12 + ROI，BPU 硬件抠图 resize
  dnn_node_para_ptr_->model_task_type =
      hobot::dnn_node::ModelTaskType::ModelRoiInferType;
  dnn_node_para_ptr_->task_num = task_num_;
  return 0;
}

void MonoOcrRecognitionNode::RosImgProcess(
    const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  if (!msg || !rclcpp::ok()) return;
  // 缓存整图 NV12 pyramid（resizer 模型输入）
  if (msg->encoding != "nv12" && msg->encoding != "nv12_8uc1") {
    RCLCPP_WARN_ONCE(rclcpp::get_logger("mono_ocr_recognition"),
                     "Unsupported img encoding: %s", msg->encoding.c_str());
    return;
  }
  auto pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
      reinterpret_cast<const char *>(msg->data.data()), msg->height,
      msg->width, msg->height, msg->width);
  if (!pyramid) {
    RCLCPP_ERROR(rclcpp::get_logger("mono_ocr_recognition"),
                 "GetNV12PyramidFromNV12Img fail");
    return;
  }
  std::lock_guard<std::mutex> lk(cache_img_mtx_);
  cache_pyramid_ = pyramid;
  cache_img_h_ = msg->height;
  cache_img_w_ = msg->width;
  has_cache_img_ = true;
}

#ifdef SHARED_MEM_ENABLED
void MonoOcrRecognitionNode::SharedMemImgProcess(
    const hbm_img_msgs::msg::HbmMsg2160P::ConstSharedPtr msg) {
  if (!msg || !rclcpp::ok()) return;

  std::stringstream ss;
  ss << "Recved img encoding: "
     << std::string(reinterpret_cast<const char *>(msg->encoding.data()))
     << ", h: " << msg->height << ", w: " << msg->width
     << ", step: " << msg->step << ", index: " << msg->index
     << ", stamp: " << msg->time_stamp.sec << "_"
     << msg->time_stamp.nanosec << ", data size: " << msg->data_size;
  RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
  
  ProcessSharedMemNv12(reinterpret_cast<const uint8_t *>(msg->data.data()),
                       msg->height, msg->width,
                       std::string(reinterpret_cast<const char *>(msg->encoding.data())));
}

void MonoOcrRecognitionNode::ProcessSharedMemNv12(const uint8_t *data,
                                                  uint32_t msg_h,
                                                  uint32_t msg_w,
                                                  const std::string &encoding) {
  if (!rclcpp::ok()) return;
  if ("nv12" != encoding) {
    RCLCPP_WARN_ONCE(rclcpp::get_logger("mono_ocr_recognition"),
                     "Unsupported shared mem encoding: %s", encoding.c_str());
    return;
  }
  // 4K 模式：det 的 ROI 是全帧坐标，而 resizer 限制 ROI 宽 ≤320*3.5=1120px；
  // 用半分辨率 pyramid（宽 ≤2336）使所有 ROI 落在限内，ROI 坐标在
  // AiMsgProcess 里按比例换算。文本行高仍远大于 rec 输入高度 48，质量无损。
  cv::Mat nv12_full(msg_h * 3 / 2, msg_w, CV_8UC1,
                    const_cast<uint8_t *>(data));
  cv::Mat pyramid_src = nv12_full;
  uint32_t py_h = msg_h, py_w = msg_w;
  if (msg_w > 2336) {
    py_w = 2336;
    py_h = static_cast<uint32_t>(msg_h * 2336.0 / msg_w) & ~1;
    cv::Mat y = nv12_full(cv::Rect(0, 0, msg_w, msg_h));
    cv::Mat uv = nv12_full(cv::Rect(0, msg_h, msg_w, msg_h / 2));
    cv::Mat dy, duv, dnv;
    cv::resize(y, dy, cv::Size(py_w, py_h));
    cv::resize(uv, duv, cv::Size(py_w, py_h / 2));
    dnv.create(py_h * 3 / 2, py_w, CV_8UC1);
    dy.copyTo(dnv(cv::Rect(0, 0, py_w, py_h)));
    duv.copyTo(dnv(cv::Rect(0, py_h, py_w, py_h / 2)));
    pyramid_src = dnv;
  }
  auto pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
      reinterpret_cast<const char *>(pyramid_src.data), py_h, py_w, py_h, py_w);
  if (!pyramid) {
    RCLCPP_ERROR(rclcpp::get_logger("mono_ocr_recognition"),
                 "GetNV12PyramidFromNV12Img fail");
    return;
  }
  std::lock_guard<std::mutex> lk(cache_img_mtx_);
  cache_pyramid_ = pyramid;
  cache_img_h_ = py_h;
  cache_img_w_ = py_w;
  full_frame_h_ = msg_h;
  full_frame_w_ = msg_w;
  cache_full_ = nv12_full.clone();
  has_cache_img_ = true;
}
#endif

void MonoOcrRecognitionNode::AiMsgProcess(
    const ai_msgs::msg::PerceptionTargets::ConstSharedPtr msg) {
  if (!msg || !rclcpp::ok()) return;

  // 检查是否有缓存图像（整图 NV12）
  cv::Mat full_frame;
  int full_h = 0, full_w = 0;
  {
    std::lock_guard<std::mutex> lk(cache_img_mtx_);
    if (!has_cache_img_ || cache_full_.empty()) {
      RCLCPP_WARN(rclcpp::get_logger("mono_ocr_recognition"),
                  "No cached image for det msg, skip");
      return;
    }
    full_frame = cache_full_;
    full_h = full_frame_h_;
    full_w = full_frame_w_;
  }

  // 软件识别路径：逐框从全帧 NV12 裁剪 -> 等比缩放(高48) -> 右侧补黑到模型宽，
  // 精确复刻 PaddleOCR rec 预处理。resizer 整块拉伸会把长行压扁 ~2.5 倍导致
  // 宽行乱码（float 对照实验：等比+补白后宽行全部读对），本路径消除该失真。
  // 每框一个 640x48 pyramid + 恒等 ROI [0,0,640,48]。
  const int kModelW = model_input_width_;
  const int kModelH = model_input_height_;
  auto rois = std::make_shared<std::vector<hbDNNRoi>>();
  std::vector<cv::Mat> crop_bufs;  // 保持 buffer 存活到同步 Run 结束
  std::vector<std::shared_ptr<hobot::dnn_node::NV12PyramidInput>> crop_pyrs;
  int text_count = 0;
  for (const auto &target : msg->targets) {
    for (const auto &roi : target.rois) {
      if (roi.type != "text") continue;
      // det 框是全帧坐标；偶对齐（NV12 UV 采样要求）
      int x = static_cast<int>(roi.rect.x_offset) & ~1;
      int y = static_cast<int>(roi.rect.y_offset) & ~1;
      int right = std::min(static_cast<int>(roi.rect.x_offset + roi.rect.width), full_w) & ~1;
      int bottom = std::min(static_cast<int>(roi.rect.y_offset + roi.rect.height), full_h) & ~1;
      x = std::max(0, x); y = std::max(0, y);
      int cw = right - x, ch = bottom - y;
      cv::Mat nv12;
      int nw = (cw >= 8 && ch >= 8)
          ? std::max(8, std::min(kModelW,
              static_cast<int>(std::round(cw * static_cast<double>(kModelH) / ch))))
          : 0;
      bool vps_ok = false;
      if (nw >= 8 && !full_frame.empty()) {
        // VPS 硬件 crop+resize（等比）：整帧 NV12 一次调用，替代 CPU
        // 裁剪+双平面 resize，省出 CPU 给 det 后处理/推理调度
        cv::Mat cropped = hobot_cv::hobotcv_crop(
            full_frame, full_h, full_w, kModelH, nw,
            cv::Range(y, y + ch), cv::Range(x, x + cw), hobot_cv::HOBOTCV_VPS);
        if (!cropped.empty() && cropped.rows >= kModelH * 3 / 2 &&
            cropped.cols >= nw) {
          nv12.create(kModelH * 3 / 2, kModelW, CV_8UC1);
          cropped(cv::Rect(0, 0, nw, kModelH * 3 / 2))
              .copyTo(nv12(cv::Rect(0, 0, nw, kModelH * 3 / 2)));
          // 右侧补黑（Y=0, UV=128 中性灰）
          if (nw < kModelW) {
            nv12(cv::Rect(nw, 0, kModelW - nw, kModelH)).setTo(0);
            nv12(cv::Rect(nw, kModelH, kModelW - nw, kModelH / 2)).setTo(128);
          }
          vps_ok = true;
        }
      }
      if (!vps_ok && cw >= 8 && ch >= 8 && !full_frame.empty()) {
        // CPU 兜底：VPS 失败或尺寸不支持时
        cv::Mat y_c = full_frame(cv::Rect(x, y, cw, ch)).clone();
        cv::Mat uv_c = full_frame(cv::Rect(x, full_h + y / 2, cw, ch / 2)).clone();
        cv::Mat y_r, uv_r;
        cv::resize(y_c, y_r, cv::Size(nw, kModelH), 0, 0, cv::INTER_LINEAR);
        cv::resize(uv_c, uv_r, cv::Size(nw, kModelH / 2), 0, 0, cv::INTER_LINEAR);
        nv12.create(kModelH * 3 / 2, kModelW, CV_8UC1);
        y_r.copyTo(nv12(cv::Rect(0, 0, nw, kModelH)));
        uv_r.copyTo(nv12(cv::Rect(0, kModelH, nw, kModelH / 2)));
        if (nw < kModelW) {
          nv12(cv::Rect(nw, 0, kModelW - nw, kModelH)).setTo(0);
          nv12(cv::Rect(nw, kModelH, kModelW - nw, kModelH / 2)).setTo(128);
        }
      } else if (!vps_ok) {
        // 退化框：全黑占位，保持 roi 与输出顺序对齐（解码为空文本）
        nv12.create(kModelH * 3 / 2, kModelW, CV_8UC1);
        nv12(cv::Rect(0, 0, kModelW, kModelH)).setTo(0);
        nv12(cv::Rect(0, kModelH, kModelW, kModelH / 2)).setTo(128);
      }
      crop_bufs.push_back(nv12);
      auto pyr = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
          reinterpret_cast<const char *>(nv12.data), kModelH, kModelW, kModelH, kModelW);
      if (!pyr) {
        RCLCPP_ERROR(rclcpp::get_logger("mono_ocr_recognition"),
                     "GetNV12PyramidFromNV12Img fail for crop %d", text_count);
        return;
      }
      crop_pyrs.push_back(pyr);
      rois->emplace_back(0, 0, kModelW, kModelH);
      text_count++;
    }
  }

  if (rois->empty()) {
    // 无文本时也发布空结果：websocket 依赖 smart 消息打开图像推送门控，
    // USB 实时场景无文字是常态，静默不发布会让 web 整体黑屏
    ai_msgs::msg::PerceptionTargets::UniquePtr empty_pub(
        new ai_msgs::msg::PerceptionTargets());
    empty_pub->set__header(msg->header);
    empty_pub->set__fps(msg->fps);
    empty_pub->set__perfs(msg->perfs);
    msg_publisher_->publish(std::move(empty_pub));
    return;
  }

  auto dnn_output = std::make_shared<MonoOcrRecognitionOutput>();
  dnn_output->model_h = model_input_height_;
  dnn_output->model_w = model_input_width_;
  dnn_output->det_msg = msg;
  dnn_output->roi_count = text_count;

  // inputs 数量必须等于 ROI 数量；每个 ROI 对应各自的 640x48 裁剪 pyramid
  std::vector<std::shared_ptr<DNNInput>> inputs;
  for (size_t i = 0; i < rois->size(); i++) {
    inputs.push_back(crop_pyrs[i]);
  }
  // 一次批量推理所有 ROI（同步），PostProcess 里逐个 CTC 解码
  int ret = Run(inputs, dnn_output, rois, true);
  if (ret != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("mono_ocr_recognition"),
                 "Run failed, ret=%d", ret);
  }
}

int MonoOcrRecognitionNode::PostProcess(
    const std::shared_ptr<DnnNodeOutput> &node_output) {
  if (!rclcpp::ok()) return -1;
  auto pout = std::dynamic_pointer_cast<MonoOcrRecognitionOutput>(node_output);
  if (!pout || pout->output_tensors.empty()) {
    RCLCPP_ERROR(rclcpp::get_logger("mono_ocr_recognition"),
                 "Invalid output");
    return -1;
  }
  if (!pout->det_msg || !msg_publisher_) return -1;

  // 批量 ROI 输出：output_tensors[i] = 第 i 个 ROI 的 CTC 输出
  // 逐个 CTC 解码得到文字
  auto post_start = std::chrono::steady_clock::now();
  std::vector<std::string> texts;
  for (auto &tensor : pout->output_tensors) {
    std::string text = mono_ocr_recognition::CtcDecode(tensor,
                                              pout->model_w / 8);
    texts.push_back(text);
    RCLCPP_INFO(rclcpp::get_logger("mono_ocr_recognition"),
                "Recognized text: %s", text.c_str());
  }
  int post_ms = static_cast<int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - post_start).count());

  // 把识别文字填进 roi.type，统一发布
  ai_msgs::msg::PerceptionTargets::UniquePtr pub_data(
      new ai_msgs::msg::PerceptionTargets());
  pub_data->set__header(pout->det_msg->header);
  pub_data->set__fps(pout->det_msg->fps);
  pub_data->set__perfs(pout->det_msg->perfs);

  size_t idx = 0;
  // 4K 模式下发布坐标换算到 web 显示空间：
  //   fb 回灌：显示半分辨率副本 -> ×(cache/full)；usb 4K 相机：显示全分辨率帧 -> ×1.0
  double out_scale = 1.0;
  if (full_frame_w_ > 0 && !display_full_resolution_) {
    out_scale = static_cast<double>(cache_img_w_) / full_frame_w_;
  }
  // 算法运行情况日志：识别批大小 + 推理/后处理耗时 + 检测到识别的链路延迟
  if (node_output->rt_stat && node_output->rt_stat->fps_updated) {
    auto now_tp = std::chrono::system_clock::now();
    auto det_tp = std::chrono::system_clock::time_point(
        std::chrono::seconds(pout->det_msg->header.stamp.sec) +
        std::chrono::nanoseconds(pout->det_msg->header.stamp.nanosec));
    int det_delay_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now_tp - det_tp).count());
    RCLCPP_WARN(this->get_logger(),
                "Rec rois: %zu, Sub fps: %.2f, Smart fps: %.2f, "
                "det msg delay ms: %d, infer time ms: %d, "
                "post process time ms: %d",
                pout->output_tensors.size(),
                node_output->rt_stat->input_fps,
                node_output->rt_stat->output_fps,
                det_delay_ms,
                node_output->rt_stat->infer_time_ms,
                post_ms);
  }

  for (const auto &target : pout->det_msg->targets) {
    ai_msgs::msg::Target t;
    t.set__type(target.type);
    t.set__track_id(target.track_id);
    for (const auto &roi : target.rois) {
      ai_msgs::msg::Roi r = roi;
      if (out_scale != 1.0) {
        r.rect.x_offset = static_cast<uint32_t>(roi.rect.x_offset * out_scale);
        r.rect.y_offset = static_cast<uint32_t>(roi.rect.y_offset * out_scale);
        r.rect.width = static_cast<uint32_t>(roi.rect.width * out_scale);
        r.rect.height = static_cast<uint32_t>(roi.rect.height * out_scale);
      }
      if (roi.type == "text" && idx < texts.size()) {
        r.set__type(texts[idx]);
        idx++;
      }
      t.rois.emplace_back(r);
    }
    pub_data->targets.emplace_back(t);
  }
  msg_publisher_->publish(std::move(pub_data));
  return 0;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(MonoOcrRecognitionNode)
