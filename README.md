# mono_ocr_recognition

English | [简体中文](README_cn.md)

mono_ocr_recognition algorithm node based on dnn_node. JSON config driven, same pattern as
`mono_dosod26`.

## Configuration

`config/mono_ocr_recognition_workconfig.json` (edit to match your model / post-process):

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

> TODO: replace placeholder values (model_output_count / reg_max / strides /
> score_threshold / nms_threshold / nms_top_k / output_order) with your
> algorithm's real values. 

## Build

RDK S100: `bash robot_dev_config/build.sh -p S100 -s mono_ocr_recognition`.

## Run

- Camera mode (mipi)

```shell
export CAM_TYPE=mipi
source /opt/ros/humble/setup.bash
source ./install/setup.bash
ros2 launch mono_ocr_recognition mono_ocr_recognition.launch.py
```

- Camera mode (usb)

```shell
export CAM_TYPE=usb
source /opt/ros/humble/setup.bash
source ./install/setup.bash
ros2 launch mono_ocr_recognition mono_ocr_recognition.launch.py
```

- Image feedback mode

```shell
export CAM_TYPE=fb
source /opt/ros/humble/setup.bash
source ./install/setup.bash
ros2 launch mono_ocr_recognition mono_ocr_recognition.launch.py
```

## Node ROS parameters

| Parameter | Default | Description |
|---|---|---|
| `config_file` | `config/mono_ocr_recognition_workconfig.json` | JSON config file |
| `feed_type` | `1` | 0: local image, 1: subscribe image |
| `is_shared_mem_sub` | `0` | subscribe `/hbmem_img` via shared memory |
| `ai_msg_pub_topic_name` | `/hobot_ocr_recognition` | AI message topic |
