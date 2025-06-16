
# Kibo RPC 6 - Image Processing & Model Training Repository

This repository contains all related files, code, and datasets used during participation in the **6th Kibo Robot Programming Challenge (Kibo RPC)**. The focus of this repository is on **image processing**, **object detection**, **Aruco marker detection**, and **YOLO model training** specifically tailored for the Kibo RPC space simulation environment.

---

## 🛰️ Project Overview

The Kibo RPC is an international programming challenge organized by JAXA and other space agencies. This repo includes a full workflow for visual data analysis and decision-making modules using image-based input from the simulated environment.

---

## 🔍 Key Features

- **YOLO Training and Inference**  
  Scripts and configuration to train a YOLO model for custom object detection tasks within the Kibo simulation.

- **ArUco Marker Detection**  
  Accurate detection and angle estimation of ArUco markers using OpenCV.

- **Cartesian Cropping**  
  Custom cropping methods using reference coordinates to isolate target objects.

- **Image Thresholding**  
  Utilities to convert images for binary classification, pre-processing, or region highlighting.

- **Dataset Creation**  
  Tools to label, crop, and generate training-ready datasets from raw images.

---

## 📦 Dependencies

- Python 3.8+
- OpenCV
- NumPy
- PyTorch (for YOLO)
- `ultralytics` (depending on the YOLO version used)

---

## 📸 Example Outputs

*(See `model_train/my_trained_models/yolo11s_300_480_grayscale/` folder)*

---

## 🧠 Contributions

This repository is maintained by [CoU Elementa]. Contributions, forks, or issues are welcome.

---

## 📄 License

MIT License

---