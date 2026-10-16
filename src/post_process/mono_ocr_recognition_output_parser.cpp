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

#include "include/post_process/mono_ocr_recognition_output_parser.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "rapidjson/document.h"
#include "rclcpp/rclcpp.hpp"

#include "dnn_node/util/output_parser/utils.h"

namespace mono_ocr_recognition {

// CTC 相关常量
static const int kTimeSteps = 40;  // rec 输出时间步数
static const int kBlankIndex = 0;  // blank 索引
static int g_num_chars = 18708;    // 字符字典字符数（不含 blank/space）
static int g_space_index = 0;      // space 索引 = num_chars + 1

// 字符字典：index 0 = blank, 1..N = 字符, N+1 = space
static std::vector<std::string> g_char_dict;

int LoadConfig(const rapidjson::Document &document) {
  if (document.HasMember("dict_num_chars"))
    g_num_chars = document["dict_num_chars"].GetInt();
  g_space_index = g_num_chars + 1;
  return 0;
}

int LoadCharacterDict(const std::string &dict_path) {
  std::ifstream ifs(dict_path);
  if (!ifs) {
    RCLCPP_ERROR(rclcpp::get_logger("mono_ocr_recognition_parser"),
                 "Read char dict fail: %s", dict_path.c_str());
    return -1;
  }
  g_char_dict.clear();
  g_char_dict.push_back("<blank>");  // index 0
  std::string line;
  while (std::getline(ifs, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    g_char_dict.push_back(line);
  }
  g_num_chars = static_cast<int>(g_char_dict.size()) - 1;
  g_space_index = g_num_chars + 1;
  RCLCPP_INFO(rclcpp::get_logger("mono_ocr_recognition_parser"),
              "Loaded char dict: %d chars", g_num_chars);
  return 0;
}

std::string CtcDecode(
    const std::shared_ptr<hobot::dnn_node::DNNTensor> &tensor, int time_steps) {
  if (!tensor) return "";
  if (time_steps <= 0) time_steps = kTimeSteps;

  tensor->CACHE_INVALIDATE();

  // rec 输出 [1, 40, 18710] float32（CTC softmax 概率）
  float *data = tensor->GetTensorData<float>();
  if (!data) return "";

  int vocab = g_num_chars + 2;  // 18710 = 18708 + blank + space

  // argmax 每时间步 + CTC 贪心解码
  std::string result;
  int prev = -1;
  for (int t = 0; t < time_steps; t++) {
    int argmax = 0;
    float max_val = data[t * vocab];
    for (int c = 1; c < vocab; c++) {
      float v = data[t * vocab + c];
      if (v > max_val) {
        max_val = v;
        argmax = c;
      }
    }
    if (argmax == kBlankIndex) {
      prev = -1;
      continue;
    }
    if (argmax != prev) {
      if (argmax == g_space_index) {
        result += ' ';
      } else if (argmax > 0 && argmax <= g_num_chars &&
                 argmax < static_cast<int>(g_char_dict.size())) {
        result += g_char_dict[argmax];
      }
    }
    prev = argmax;
  }
  return result;
}

int32_t Parse(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output,
    std::shared_ptr<DnnParserResult> &result) {
  // rec 结果不走标准 Perception，node 层直接调 CtcDecode。此处空实现。
  if (!result) result = std::make_shared<DnnParserResult>();
  return 0;
}

}  // namespace mono_ocr_recognition
