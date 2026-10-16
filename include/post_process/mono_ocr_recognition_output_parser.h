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

#ifndef MONO_OCR_RECOGNITION_OUTPUT_PARSER_H_
#define MONO_OCR_RECOGNITION_OUTPUT_PARSER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rapidjson/document.h"

#include "dnn_node/dnn_node_data.h"
#include "dnn_node/util/output_parser/perception_common.h"

using hobot::dnn_node::output_parser::Bbox;
using hobot::dnn_node::output_parser::Detection;
using hobot::dnn_node::output_parser::DnnParserResult;
using hobot::dnn_node::output_parser::Perception;

namespace mono_ocr_recognition {

// 从 workconfig.json 读取配置（含字符字典路径）。
int LoadConfig(const rapidjson::Document &document);

// 加载字符字典文件（每行一个字符，UTF-8 编码）。
// 返回 0 成功，-1 失败。
int LoadCharacterDict(const std::string &dict_path);

// 对 CTC 输出做贪心解码，输出识别文字。
// tensor: rec 模型输出 [1, 40, 18710] float32（CTC softmax 概率）
// 返回识别出的文字（UTF-8）。
std::string CtcDecode(
      const std::shared_ptr<hobot::dnn_node::DNNTensor> &tensor,
      int time_steps = 0);

// 供 node 层调用：对单个文本行的输出 tensor 解码出文字。
// 保持与 det parser 类似的 Parse 签名（但 rec 结果不走标准 Perception）。
int32_t Parse(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output,
    std::shared_ptr<DnnParserResult> &result);

}  // namespace mono_ocr_recognition
#endif  // MONO_OCR_RECOGNITION_OUTPUT_PARSER_H_
