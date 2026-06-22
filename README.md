# Pneumonia Detection — Chest X-Ray Classification System
 
**AAI-540-Machine Learning Operations**
**University of San Diego — Shiley-Marcos School of Engineering**
 
Spencer Cody · Aliaksei Matsarksi · Birendra Khimding
 
---
 
## Project Overview
 
An end to end MLOps pipeline that detects pneumonia from chest X-ray images using a Convolutional Neural Network (CNN) deployed on AWS SageMaker. 
 
The model classifies X-ray images as **Normal** or **Pneumonia** using a 4-block CNN trained on 33,000 images from two public datasets.
 
## Datasets
 
| Dataset | Format | Images | Source |
|---|---|---|---|
| [Chest X-Ray Images (Kermany)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data) | JPEG | 5,863 | Kaggle |
| [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data) | DICOM | 29,700+ | Kaggle |
 
Combined total: 33,720 images (22,255 Normal / 11,465 Pneumonia)
 
## Architecture
 
<img width="850" height="1971" alt="pneumonia_architecture" src="https://github.com/user-attachments/assets/10a39263-c0e4-437a-a829-5e7a593930d7" />



 
## AWS Services Used
 
- **S3** — Raw and preprocessed image storage (data lake)
- **Amazon Athena** — SQL queries over image metadata for exploration
- **SageMaker Studio** — Notebooks for development, training, and evaluation
- **SageMaker Pipelines** — Automated CI/CD pipeline (split → train → evaluate → quality gate)
- **SageMaker Feature Store** — Queryable training manifest (online + offline stores)
- **SageMaker Model Registry** — Versioned model catalog with approval workflow
- **SageMaker Endpoint** — Real time inference via ONNX Runtime
- **SageMaker Model Monitor** — Model quality and data quality baselines
- **CloudWatch** — Infrastructure dashboard and  alarms
## Model
 
- **Architecture:** 4-block CNN (filters 32 → 64 → 128 → 256), each block has two Conv2D layers with BatchNormalization, ReLU, and MaxPooling
- **Input:** 128×128 grayscale images
- **Training:** batch size 16, learning rate 1e-3, up to 10 epochs, Adam optimizer, binary cross-entropy loss
- **Data Augmentation:** random flip, rotation, zoom, contrast, translation (training set only)
- **Class Weights:** inverse-frequency weighting to handle ~2:1 Normal to Pneumonia imbalance
- **Classification Threshold:** 0.4 (lowered from 0.5 to prioritize recall — missed pneumonia is more dangerous than a false alarm)
- **Deployment:** Keras model converted to ONNX format (tf2onnx) due to SageMaker container compatibility issues, deployed with custom inference script on SKLearn container

 
## Repository Structure
 
```
├── config.py                    # Shared configuration (bucket name, image size)
├── Data_Setup.ipynb             # Download raw datasets and upload to S3
├── Data_Preparations.ipynb      # Preprocess images, Feature Store, Athena catalog
├── EDA.ipynb                    # Exploratory data analysis (class balance, pixel stats)
├── CNN_Model.ipynb              # CNN development, training, and evaluation
├── Model_Monitoring.ipynb       # ONNX deployment, endpoint, CloudWatch, Model Monitor
├── CICD_Pipeline.ipynb          # SageMaker Pipelines CI/CD (direct S3 read variant)      
```
 


 
## CI/CD Pipeline (SageMaker Pipelines)
 
The pipeline automates the training workflow as a directed acyclic graph (DAG):
 
1. **PneumoniaSplit** — Stratified 40/10/10/40 split with class weight computation
2. **PneumoniaTrain** — 4-block CNN training on TensorFlow container
3. **PneumoniaEvaluate** — Compute accuracy, precision, recall, F1, AUC at threshold 0.4
4. **PneumoniaQualityGate** — If F1 ≥ 0.70 → register model; if not → stop pipeline
All parameters (instance type, epochs, batch size, F1 threshold, approval status) are tunable per run without code changes.
 
## Monitoring
 
- **CloudWatch Dashboard:** Pneumonia-CNN-Monitoring — tracks invocation count, latency, CPU, memory, disk, and custom model metrics
- **CloudWatch Alarms:** high error rate, high latency, low invocations, high CPU, high memory, accuracy drop
- **Model Quality Baseline:** 3,000 labeled images — expected accuracy, precision, recall, F1, AUC
- **Data Quality Baseline:** 2,000 images — pixel intensity distributions (mean, std, min, max, median)
## Setup
 
 
### Install Dependencies
 
Each notebook installs its own dependencies at the top. Core libraries:
 
sagemaker>=2.0,<3.0
boto3
tensorflow
pydicom
opencv
pyathena
awswrangler
tf2onnx
onnxruntime
