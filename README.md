# Duality AI's Space Station Challenge: Safety Object Detection

**Team Name:** [**Your Team Name Here**]
**Final mAP@0.5 Score:** **72.6%**

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)
![Model](https://img.shields.io/badge/Model-YOLOv8s-violet.svg)

---

## 📋 Table of Contents

1.  [**Project Overview**](#-project-overview)
2.  [**Key Features**](#-key-features)
3.  [**Final Results & Performance**](#-final-results--performance)
4.  [**In-Depth Model Analysis**](#-in-depth-model-analysis)
    * [Performance by Class](#performance-by-class)
    * [Failure Case Analysis](#failure-case-analysis)
5.  [**Technology Stack**](#-technology-stack)
6.  [**Setup and Installation**](#-setup-and-installation)
7.  [**Usage Instructions**](#-usage-instructions)
    * [Running Predictions](#running-predictions)
    * [Re-training the Model](#re-training-the-model)
8.  [**Project File Structure**](#-project-file-structure)
9.  [**(Bonus) Use Case Application**](#bonus-use-case-application)

---

## 📝 Project Overview

This project is a submission for the Duality AI's Space Station Challenge. The core task was to train a high-performing object detection model to identify 7 critical safety items in a synthetic space station environment. Ensuring the correct identification of equipment like oxygen tanks and fire extinguishers is a mission-critical capability for ensuring operational safety and crew well-being in isolated environments like a space station.

We utilized the provided synthetic dataset of over 1500 images generated from Duality AI's Falcon platform, which included challenging scenarios with varied lighting, object occlusions, and diverse camera angles. Our approach involved using the YOLOv8s architecture, carefully monitoring the training process to achieve optimal performance, and conducting a thorough analysis of the model's strengths and weaknesses.

And additionally to further enhance the model to detect the objects in complex scenarios, we have also augumented the data in several conditions(like more image noise, less contrast, more contrast, bright vs dim image etc). 

---

## ✨ Key Features

* **High-Performance Model:** Achieved a final score of **72.6% mAP@0.5**, significantly outperforming the 40-50% baseline.
* **Robust Detection:** The model demonstrates strong performance in identifying key objects like `OxygenTank` and `SafetySwitchPanel` with high precision.
* **Detailed Error Analysis:** A comprehensive analysis of model failures, including missed detections and misclassifications, providing a clear path for future improvements.
* **Reproducible Workflow:** The project is structured with clear scripts and instructions, allowing for easy reproduction of our final results.

---

## 🏆 Final Results & Performance

Our model was evaluated on the provided test dataset. The key performance metric is Mean Average Precision at an IoU threshold of 0.5 (mAP@0.5).

**Overall mAP@0.5: 72.6%**

![Precision-Recall Curve](scripts/runs/detect/val/BoxPR_curve.png)
_The Precision-Recall curve shows the model's performance across all classes, with the overall mAP score highlighted._

### Per-Class mAP Scores

| Class                 | mAP@0.5 Score |
| --------------------- | :-----------: |
| **OxygenTank** |     0.802     |
| **NitrogenTank** |     0.785     |
| **SafetySwitchPanel** |     0.755     |
| **FirstAidBox** |     0.754     |
| **FireAlarm** |     0.686     |
| **FireExtinguisher** |     0.671     |
| **EmergencyPhone** |     0.628     |

---

## 🔬 In-Depth Model Analysis

A deeper analysis of the results reveals specific areas where the model excels and struggles.

![Normalized Confusion Matrix](scripts/runs/detect/val/confusion_matrix_normalized.png)
_The normalized confusion matrix visualizes the model's per-class accuracy and its specific misclassifications._

### Performance by Class

* **Strong Performance:** The model shows high accuracy for `SafetySwitchPanel` (76%), `OxygenTank` (74%), and `NitrogenTank` (72%). These objects have distinct features that the model learned effectively.
* **Areas for Improvement:** The model had the most difficulty with `FireExtinguisher` (59%) and `EmergencyPhone` (62%). These objects may have appeared in more challenging lighting conditions or had more varied appearances in the dataset.

### Failure Case Analysis

We analyzed specific prediction errors to understand the model's limitations.

* **False Negatives (Missed Detections):** The most common error was the model failing to detect an object and classifying it as background. The confusion matrix shows this was a particular issue for `EmergencyPhone` (38% missed) and `FireExtinguisher` (37% missed). This can happen in low-light conditions or when an object is heavily occluded.
    * **Example:** ![Missed Objects](scripts/predictions/images/000000136_vlight_uncluttered.png)
    * ![Missed Objects]<img width="1410" height="866" alt="Screenshot 2025-09-21 193642" src="https://github.com/user-attachments/assets/47b7dee5-8ff3-4ccc-aac7-b06094808e77" />
    * Here as you can see in the first image the Objects which were in the focus point have been detected with high accuracy, but as seen in the second image 2 objects(First Aid Box and a Emergency Phone) have been completely missed.
* **Misclassifications:** While less common, the model occasionally confused similar-looking objects.
    * **Example:** A `FirstAidBox` was sometimes misclassified as a `FireAlarm` (5% of the time), likely due to both being red and box-shaped. _[If you find an example image, insert it here.]_

---

## 💻 Technology Stack

* **Python 3.10**
* **PyTorch**
* **Ultralytics YOLOv8**
* **OpenCV**
* **PyYAML**

---

## ⚙️ Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone [your-repo-link]
    cd [your-repo-name]
    ```

2.  **Create Conda Environment:** It is recommended to use a Conda environment to manage dependencies.
    ```bash
    conda create --name duality-hackathon python=3.10 -y
    conda activate duality-hackathon
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You may need to create a `requirements.txt` file containing `ultralytics`, `opencv-contrib-python`, and `pyyaml`)*

---

## 🚀 Usage Instructions

### Running Predictions

To evaluate our final model and reproduce the **72.6% mAP@0.5** score, navigate to the `scripts` directory and run `predict.py`.

**Note:** Ensure the `runs` directory from our submission is located at the root of the project folder, one level above the `scripts` directory.

```bash
cd scripts
python predict.py
```

The script will output the final metrics to the console and save prediction images in the `scripts/predictions/` folder.

### Re-training the Model

The full training process can be replicated by running the `train.py` script from the `scripts` directory.

```bash
cd scripts
python train.py --epochs 100 --batch 8 --mosaic 0.5 --optimizer AdamW --lr0 0.001
```

---

## 📁 Project File Structure

```
.
├── Duality_Hackathon/
│   ├── dataset/
│   │   ├── train/
│   │   └── val/
│   ├── runs/
│   │   └── train/
│   │       └── exp2/
│   │           ├── weights/
│   │           │   └── best.pt   <-- Our final trained model
│   │           └── ... (all result graphs)
│   ├── scripts/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── yolo_params.yaml
│   └── README.md
└── ...
```

---

## ✨ (Bonus) Use Case Application

_[**This is a placeholder for your bonus application. Describe your app in detail here.**]_

**1. Application Concept:**
We developed a proof-of-concept monitoring application using Python and OpenCV. The application processes a video stream and uses our trained YOLOv8s model to detect and flag safety equipment in real-time. A visual overlay is drawn on the video feed, highlighting each detected object with a bounding box and a confidence score.

**2. Implementation Details:**
The application is built as a single Python script. It loads the `best.pt` model weights and uses a `while` loop to read frames from a video file. Each frame is passed to the model for inference. The results are then used to draw visualizations on the frame before it's displayed on the screen.

**3. Model Maintenance with Falcon:**
To keep the model up-to-date in a real-world scenario, Duality AI's Falcon platform would be essential. If a new piece of equipment is added to the space station, or existing equipment is redesigned, we could:
1.  **Update the Digital Twin:** Add or modify the 3D models of the equipment within the Falcon simulation.
2.  **Generate New Synthetic Data:** Use Falcon to generate thousands of new, perfectly labeled training images of the new/modified equipment under diverse conditions (lighting, angles, occlusions).
3.  **Fine-tune the Model:** Use this new synthetic data to quickly fine-tune our existing model, teaching it to recognize the new objects without requiring costly and time-consuming real-world data collection.

**[Link to Video Demonstration]**
