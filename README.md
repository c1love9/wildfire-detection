# Wildfire Detection

> **实习项目 Fork：智能火灾报警系统**
>
> 本仓库在保留上游来源和许可证的基础上逐步改进。当前支持图片检测、逐帧视频与摄像头实时检测、连续确认、分级报警、持久化日志和报警截图／片段回放。
>
> - [本地启动与重新安装](docs/quick_start.md)
> - [上游来源与本阶段范围](docs/upstream.md)
> - [本机复现记录](docs/reproduction.md)
> - [视频检测使用与验证记录](docs/video_detection.md)
> - [连续确认、分级报警与日志](docs/alarm.md)
> - [报警截图、前后片段与回放](docs/recording.md)
> - [摄像头实时检测与验证](docs/camera.md)
>
> Windows 本机双击 `启动项目.cmd`，然后打开 http://127.0.0.1:8501 。在侧边栏进入 `Demo Videos`，选择视频或摄像头并点击“开始检测”；`Alarm History`可查询日志、查看截图、回放片段和导出CSV。自己的模型训练、像素掩码与模拟位置展示按后续计划推进。

Welcome to the Wildfire Detection Research repository! This repository hosts the code and resources related to our research on leveraging computer vision for fire detection. Our aim is to contribute to wildfire prevention efforts by developing and training an object detection model to accurately identify instances of fire and smoke in images.

<p align="center">
  <img src="src/intro-gif.gif" alt="gif" width = 800>
</p>

## Highlights:

- Trained YOLOv8 model using the D-Fire dataset for accurate fire and smoke detection.
- Explored different model sizes and their performance metrics.
- Developed a Streamlit app for practical demonstration of the model's capabilities.
- Investigated future implications of using computer vision in wildfire prevention strategies.

#### For a comprehensive understanding of our research journey, methodology, and results, please refer to our Medium article: [Stay Ahead of the Flames: A Comprehensive Guide to Wildfire Prevention with YOLOv8](https://medium.com/institute-of-smart-systems-and-ai/stay-ahead-of-the-flames-a-comprehensive-guide-to-wildfire-prevention-with-yolov8-3eb8edd1121a)

---

## Dataset

We utilized the [D-Fire dataset](https://github.com/gaiasd/DFireDataset), a curated collection of 21,000 labeled images, each annotated in YOLO format. The dataset focuses on fire and smoke instances, while also encompassing diverse visual cues, including non-fire images that resemble fire-like patterns.
<p align="center">
  <img src="src/multiple-examples.png" alt="D-Fire dataset examples" width = 600>
</p>
<div align="center">
  D-Fire dataset examples
</div>

We have explored many different datasets. Here is the summary:

| Dataset                                                                                                                                                                                    | Image type      | Camera                               | \# images                                                                          | Classes                                                                                             | Bounding Boxes |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------- | ------------------------------------ | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | -------------- |
| [Wildfire Detection Image Data](https://www.kaggle.com/datasets/brsdincer/wildfire-detection-image-data)                                                                                   | RGB             | Regular                              | 1,875 images                                                                       | \- fire<br>\- no fire                                                                               | no             |
| [Fire Detection Using Surveillance Camera on Roads](https://www.kaggle.com/datasets/tharakan684/urecamain)                                                                                 | RGB             | Surveillance Camera                  | 10,000 images                                                                      | \- fire<br>\- no fire                                                                               | no             |
| [FIRESENSE](https://www.kaggle.com/datasets/chrisfilo/firesense) (Videos)                                                                                                                  | RGB             | Regular, Surveillance Camera         | 11 fire videos<br>16 non-fire videos<br><br>13 smoke videos<br>13 non-smoke videos | \- fire<br>\- no fire<br>\- smoke<br>\- no smoke                                                    | no             |
| [Aerial Rescue Object Detection](https://www.kaggle.com/datasets/julienmeine/rescue-object-detection)                                                                                      | RGB             | Regular, drone                       | 29,810 images                                                                      | \- human<br>\- fire<br>\- vehicle                                                                   | yes            |
| [Fire detection dataset](https://www.kaggle.com/datasets/jimishpatel/fire-detection-dataset)                                                                                               | RGB             | Regular, Surveillance Camera         | 3,894 images                                                                       | \- fire<br>\- no fire                                                                               | no             |
| [Forest Fire](https://www.kaggle.com/datasets/kutaykutlu/forest-fire)                                                                                                                      | RGB, Gray Scale | Regular, Drone, Surveillance Camera  | 15,800 images                                                                      | \- fire<br>\- smoke                                                                                 | no             |
| [Forest Fire Images](https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images)                                                                                                  | RGB             | Regular                              | 5,000 images                                                                       | \- fire<br>\- no fire                                                                               | no             |
| [Fire Detection in YOLO format](https://www.kaggle.com/datasets/ankan1998/fire-detection-in-yolo-format)                                                                                   | RGB             | Regular                              | 500 images                                                                         | \- fire                                                                                             | yes            |
| [FLAME 2: FIRE DETECTION AND MODELING: AERIAL MULTI-SPECTRAL IMAGE DATASET](https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset) | RGB, IR         | Drone                                | 53,451 RGB<br>53,451 IR                                                            | \- fire and smoke<br>\- fire and no smoke<br>\- no fire and smoke<br>\- no fire and no smoke        | no             |
| [Forest Fire Dataset](https://www.kaggle.com/datasets/alik05/forest-fire-dataset)                                                                                                          | RGB             | Regular                              | 1,900 images                                                                       | \- fire<br>\- no fire                                                                               | no             |
| [Open Wildfire Smoke Datasets](https://github.com/aiformankind/wildfire-smoke-dataset/tree/master)                                                                                         | RGB             | Surveillance Camera                  | 2,192 images                                                                       | \- smoke                                                                                            | yes            |
| [AIDER: Aerial Image Database for Emergency Response applications](https://github.com/ckyrkou/AIDER/tree/master)                                                                           | RGB             | Aerial view, regular                 | 500 images for each disaster class<br>4,000 images for the normal class            | \- Fire/Smoke<br>\- Flood<br>\- Collapsed Building/Rubble<br>\- Traffic Accidents<br>\- Normal case | no             |
| [Furg Fire Dataset](https://github.com/steffensbola/furg-fire-dataset)                                                                                                                     | RGB             | Regular                              | 21 videos                                                                          | \- fire                                                                                             | yes            |
| [Mivia Fire Detection](https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/)                                                                                    | RGB             | Regular                              | 14 fire videos<br>17 non-fire videos                                               | \- fire<br>\- no fire                                                                               | no             |
| [FireNet](https://github.com/OlafenwaMoses/FireNET)                                                                                                                                        | RGB             | Regular                              | 500 images                                                                         | \- fire                                                                                             | yes            |
| [FIRE Dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset)                                                                                                                   | RGB             | Regular                              | 755 outdoor-fire images <br>244 nature images                                      | \- fire<br>\- no fire                                                                               | no             |
| [Fire Detection v2](https://universe.roboflow.com/yi-shing-group-limited/fire-detection-v2-yn3wz)                                                                                          | RGB             | Regular                              | 600 images                                                                         | \- scale1fire<br>\- scale2fire<br>\- scale3fire                                                     | yes            |
| [fireDetection Computer Vision Project](https://universe.roboflow.com/school-tvtyg/firedetection-xxwxc)                                                                                    | RGB             | Regular                              | 9,681 images                                                                       | \- Fire<br>\- fire                                                                                  | yes            |
| [D-Fire](https://github.com/gaiasd/DFireDataset)                                                                                                                                           | RGB             | Regular, Aerial, Surveillance Camera | 21,000 images                                                                      | \- Fire<br>\- Smoke                                                                                 | yes            |
| [Fire-Smoke-Dataset](https://github.com/DeepQuestAI/Fire-Smoke-Dataset)                                                                                                                    | RGB             | Regular                              | 3,000 images                                                                       | \- Fire<br>\- Smoke<br>\- Neutral                                                                   | no             |

---

## Model Training

We trained the YOLOv8 model by [Ultralytics](https://github.com/ultralytics/ultralytics) on the D-Fire dataset to achieve accurate fire and smoke detection. Our research not only focuses on achieving high accuracy but also on optimizing model parameters and hyperparameters to ensure efficiency and speed.

#### Training recipe can be found in a [training-recipes folder](training-recipes)

---

## Results

The trained YOLOv8 model demonstrated impressive performance on the D-Fire test dataset, with mAP@50 scores and inference time across different model sizes as follows. Evaluation was done using NVIDIA A100-SXM4-40. Resolution of input images was 640x640.

| Model Size  | mAP@50 | Inference (ms) |
|-------------|--------|----------------|
| Nano        | 0.787  |     0.422      |
| Small       | 0.798  |     0.773      |
| Medium      | 0.801  |     1.532      |
| Large       | 0.812  |     2.342      |
| Extra Large | 0.814  |     3.465      |

#### For a detailed exploration of our training process and insights, we invite you to read our comprehensive guide on Medium: [Stay Ahead of the Flames: A Comprehensive Guide to Wildfire Prevention with YOLOv8](https://medium.com/institute-of-smart-systems-and-ai/stay-ahead-of-the-flames-a-comprehensive-guide-to-wildfire-prevention-with-yolov8-3eb8edd1121a)

---

## Future Implications

This research underscores the potential of computer vision in addressing real-world challenges, such as wildfire detection. As technology evolves, integrating machine learning tools into wildfire prevention and emergency response strategies could significantly enhance our ability to detect and mitigate wildfires effectively.

---

## Streamlit App

For a practical demonstration of our research, you can interact with our Wildfire Detection App, powered by the YOLOv8 model. This app allows you to upload images and observe the model's detection capabilities in action. 

#### To experience the app, visit: [Wildfire Detection App](https://wildfire-detection.streamlit.app)

<p align="center">
  <img src="src/streamlit-merged.png" alt="Streamlit App" width = 600>
</p>
<div align="center">
  Wildfire Detection App
</div>

---

## Disclaimer

Please note that while our Streamlit app demonstrates the capabilities of our model, it is intended for demonstration purposes and may not be suitable for critical wildfire detection applications.

---
## Acknowledgment

I would like to acknowledge the Institute of Smart Systems and Artificial Intelligence ([ISSAI](https://issai.nu.edu.kz)) at Nazarbayev University for fostering an environment of innovation and research excellence. The support I received from ISSAI has been integral to the successful completion of this endeavor.

I extend my heartfelt appreciation to my supervisor, [Askat Kuzdeuov](https://www.linkedin.com/in/askatkuzdeuov/), at ISSAI, whose guidance and mentorship were indispensable to the success of this research. His expertise and support have been invaluable in shaping the direction and quality of this work.

I would also like to extend my thanks to the creators of the [D-Fire dataset](https://github.com/gaiasd/DFireDataset) for providing a valuable resource that underpins the foundation of this research. Additionally, the [Ultralytics](https://github.com/ultralytics/ultralytics) team's contribution to the YOLOv8 model has been instrumental in enabling accurate and efficient fire detection.

---

## License

This project is licensed under the [MIT License](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt).
