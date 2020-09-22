# smart_traffic_system
### 功能
目标检测及追踪，包括车辆、行人、交通灯、摩托车等类别

车牌识别

车辆实时速度检测

路口流量统计

违规行为检测

### 环境
imutils=0.5.3=pypi_0
keras=2.3.1=pypi_0
matplotlib=3.2.1=pypi_0
numpy=1.18.4=pypi_0
opencv-python=4.2.0.34=pypi_0
pillow=7.1.2=pypi_0
python=3.6.10=h7579374_2
scikit-learn=0.23.1=pypi_0
scipy=1.4.1=pypi_0
tensorboard=2.2.1=pypi_0
tensorflow=2.0.0=pypi_0
tensorflow-estimator=2.1.0=pypi_0
tensorflow-gpu=2.2.0=pypi_0

### 实现
yolov4检测+deepsort追踪

数据集：目标检测使用coco数据集训练，识别车辆等六个分类，车牌使用ccpd数据集训练

model_data和测试视频有空传网盘
