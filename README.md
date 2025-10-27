
## 🚗 Vehicle Detection Using YOLOv8

### 📘 Overview

This project implements an **Object Detection model** for recognizing and localizing different types of vehicles in images using **YOLOv8** (You Only Look Once, version 8).
It identifies multiple vehicle types — such as **Cars**, **Trucks**, **Buses**, **Motorcycles**, and **Ambulances** — by drawing bounding boxes and class labels around them.

---

### 🗂️ Dataset Structure

The dataset is organized in the following structure:

```
VehiclesDetectionDataset/
│
├── train/
│   ├── images/        # Training images
│   └── labels/        # YOLO-format labels (.txt)
│
├── valid/
│   ├── images/        # Validation images
│   └── labels/        # Labels for validation
│
└── test/
    ├── images/        # Testing images
    └── labels/        # Labels for testing
```

Each `.txt` label file follows the YOLO format:

```
class_id  x_center  y_center  width  height
```

(all values normalized between 0 and 1).

---

### 🧾 Dataset Configuration (vehicles.yaml)

```yaml
path: /kaggle/input/vehicledetection/VehiclesDetectionDataset
train: train/images
val: valid/images
test: test/images

nc: 5
names: ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']
```

---

### ⚙️ Training the Model

```python
from ultralytics import YOLO

# Load YOLOv8 model (nano version for faster training)
model = YOLO("yolov8n.pt")

# Train on dataset
model.train(
    data="vehicles.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="vehicle_yolov8"
)
```

> 📌 Tip: Increase epochs (e.g., 100) or use a larger model (`yolov8m.pt`) for higher accuracy.

---

### 🧪 Evaluation

Evaluate on the test set after training:

```python
results = model.val()
print(results)
```

**Example output:**

```
Precision: 0.703
mAP@50: 0.703
mAP@50-95: 0.471
```

✅ Interpretation:

* Precision ~0.70 → 70% of detections are correct.
* mAP@50 ~0.70 → Good overall detection performance.
* mAP@50-95 ~0.47 → Moderate consistency across stricter IoU thresholds.

---

### 🧮 Confusion Matrix & Metrics

```python
metrics = model.val()
metrics.plot()
```

This generates plots for:

* Confusion Matrix
* Precision–Recall Curve
* F1 Curve
* mAP per class


---

### 📊 Results Summary

| Metric    | Value     |
| --------- | --------- |
| Precision | **0.703** |
| mAP@50    | **0.703** |
| mAP@50–95 | **0.471** |

💡 **Remarks:**

* The model performs well for a lightweight YOLOv8n network.
* Increasing model size or epochs will likely push mAP above **0.8**.

---

### 🚀 Future Improvements

* Use **YOLOv8m/l** for higher accuracy.
* Perform **data augmentation** (brightness, rotation, flips).
* Fine-tune **confidence threshold** and **IoU NMS** values.
* Collect more labeled data for underrepresented classes.

---

### 🧠 Technologies Used

* **Python 3.11**
* **YOLOv8 (Ultralytics)**
* **OpenCV**
* **Matplotlib**
* **PyTorch**

---

### 👤 Author

**Mohamed Hussien Mansour**
Senior 1 – Computer Engineering, Ain Shams University
AI Head @ Semi Colon | Passionate about AI & Computer Vision

---
