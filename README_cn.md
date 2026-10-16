# mono_ocr_recognition

[English](README.md) | 简体中文

基于 dnn_node 的 mono_ocr_recognition 算法节点。JSON 配置驱动，与 `mono_dosod26`
保持相同的工程结构。

OCR 识别节点与 `mono_ocr_detection` 配套使用：订阅检测节点输出的文字框
（`/hobot_ocr_detection`）和原始 NV12 帧，逐框裁剪、等比缩放（高 48，右侧
补黑，精确复刻 PaddleOCR rec 预处理）后识别文字，把识别结果填入 roi 的
`type` 字段发布到 `/hobot_ocr_recognition`。

## 配置

`config/mono_ocr_recognition_workconfig.json`（按实际模型 / 后处理修改）：

```json
{
    "model_file": "./config/ppocrv6_medium_rec_int8_resizer_m_640.hbm",
    "model_name": "ppocrv6_medium_rec_int8_resizer_m_640",
    "char_dict_file": "./config/char_dict.txt",
    "model_input_width": 640,
    "model_input_height": 48,
    "task_num": 4
}
```

> TODO: `model_output_count / reg_max / strides / score_threshold /
> nms_threshold / nms_top_k / output_order` 等占位值需替换为算法真实值。

## 编译

RDK S100：
`bash robot_dev_config/build.sh -p S100 -s mono_ocr_recognition`。

## 运行

节点需与 `mono_ocr_detection` 同时运行（订阅其检测结果；4K 共享内存链下
还需要 NV12 原图用于裁剪）：

```shell
export CAM_TYPE=usb   # mipi / usb / fb
source /opt/ros/humble/setup.bash
source ./install/setup.bash
ros2 launch mono_ocr_recognition mono_ocr_recognition.launch.py
```

```shell
export CAM_TYPE=usb   # mipi / usb / fb
source /opt/ros/humble/setup.bash
source ./install/setup.bash
ros2 launch mono_ocr_recognition mono_ocr_recognition.launch.py
```

```shell
export CAM_TYPE=usb   # mipi / usb / fb
source /opt/ros/humble/setup.bash
source ./install/setup.bash
ros2 launch mono_ocr_recognition mono_ocr_recognition.launch.py
```

## 节点 ROS 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `config_file` | `config/mono_ocr_recognition_workconfig.json` | JSON 配置文件 |
| `feed_type` | `1` | 0: 本地图片，1: 订阅 |
| `is_shared_mem_sub` | `0` | 通过共享内存订阅 NV12 帧 |
| `sharedmem_msg_type` | `1080p` | 共享内存消息档位：`1080p` / `2160p` |
| `sharedmem_img_topic_name` | `/hbmem_img` | 共享内存图片话题 |
| `ai_msg_sub_topic_name` | `/hobot_ocr_detection` | 检测结果订阅话题 |
| `ai_msg_pub_topic_name` | `/hobot_ocr_recognition` | 识别结果发布话题 |
| `display_full_resolution` | `true` | 发布坐标保持全分辨率（web 显示全分辨率帧时必须为 true） |
| `is_sync_mode` | `0` | 同步模式 |
| `model_input_width` | `640` | 识别模型输入宽 |
| `model_input_height` | `48` | 识别模型输入高 |
