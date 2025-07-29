# Species Family Classification Using Deep Learning 

## Overview
This project aims to classify biological species families based on images from the Encyclopedia of Life (EOL). The dataset contains 202 classes with a strong imbalance in class distribution. The main challenge was to develop robust models capable of generalizing well despite varying image quality and limited samples for some classes.

## Dataset
- **Size:** Approximately 33.6k images across 202 families, with highly imbalanced class sizes (from 29 to 300 images per class).  
- **Characteristics:** Mainly RGB images standardized to 224x224 pixels, including some grayscale images.  
- **Preprocessing:** Outlier removal using feature extraction (CLIP/VGG16) and anomaly detection; normalization; one-hot encoding of labels.

## Approaches

### 1. Training CNN from Scratch
- Tested both sequential and functional architectures.  
- Addressed class imbalance using data augmentation and class weighting; class weights yielded improved F1 scores.  
- Hyperparameter tuning via Bayesian optimization.  
- Results: Limited performance, prone to overfitting.

### 2. Transfer Learning with Pretrained Models
- Models used: Xception, ResNet-50 V2, VGG16, EfficientNet; final focus on Xception and ResNet-50 V2.  
- Pipeline included moderate data augmentation, normalization, dense layers with dropout and batch normalization.  
- Advanced regularization strategies (early stopping, learning rate reduction).  
- Hyperparameter optimization with Weights & Biases (WandB) sweeps using Bayesian optimization.  
- Results: Macro F1 around 60%, improved robustness and reduced overfitting.

## Results and Insights
- Transfer learning was essential to achieve high performance.  
- Class imbalance handling via class weights improved metrics.  
- Outlier removal using CLIP embeddings enhanced dataset quality.  
- Models trained from scratch struggled to generalize, even with augmentations.  

## Technologies & Tools
- Python, TensorFlow, Keras  
- Weights & Biases (WandB) for experiment tracking and hyperparameter tuning  
- Data analysis: pandas, matplotlib, seaborn

## Conclusion
This project highlights the importance of transfer learning for image classification tasks involving imbalanced and noisy datasets. Careful data cleaning and robust regularization strategies are crucial to reduce overfitting and improve generalization.
