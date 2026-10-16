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

#ifndef MONO_OCR_RECOGNITION_NMS_H_
#define MONO_OCR_RECOGNITION_NMS_H_

#include <vector>

#include "dnn_node/util/output_parser/perception_common.h"

using hobot::dnn_node::output_parser::Detection;

namespace mono_ocr_recognition {

void nms(std::vector<Detection> &input, float iou_threshold,
         int top_k, std::vector<Detection> &result, bool suppress = false);

}  // namespace mono_ocr_recognition

#endif  // MONO_OCR_RECOGNITION_NMS_H_
