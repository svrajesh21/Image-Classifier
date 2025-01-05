# CIFAR-10 Image Classifier  

This project demonstrates a convolutional neural network (CNN) built using TensorFlow and Keras to classify images from the **CIFAR-10 dataset** into 10 distinct categories. The model achieves ~80% accuracy on the test dataset.  

---

## **About the Project**
This project demonstrates how to implement a CNN to classify images into the following 10 categories:  
`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, and `truck`.  

Key features:
- Data preprocessing (normalization).
- Data augmentation (rotations, flips, and shifts) to improve model generalization.
- Multi-layered CNN architecture for feature extraction and classification.
- Saved model for reuse or deployment.

  
---

## **How It Works**
1. **Data Preprocessing**:
   - Images are normalized to the range `[0, 1]` for faster training.  
2. **Data Augmentation**:
   - Random rotations, shifts, and horizontal flips are applied to improve generalization.  
3. **CNN Architecture**:
   - Multiple convolutional layers to extract features like edges and textures.  
   - Pooling layers to downsample and reduce complexity.  
   - Dropout layers to prevent overfitting.  
4. **Training**:
   - The model is trained for 50 epochs with the Adam optimizer.  
5. **Evaluation**:
   - The model achieves ~80% accuracy on the test dataset.  
---


## **Features**
- Classifies images into the following categories:  
  `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.  
- Implements data preprocessing (normalization) and data augmentation (random flips, rotations, and shifts).  
- A robust CNN architecture with dropout and batch normalization for better generalization.  
- Saves the trained model for future use (`cifar10_cnn_model.h5`).


  
---
## **How It Works**
1. **Data Preprocessing**:
   - Images are normalized to the range `[0, 1]` for faster training.  
2. **Data Augmentation**:
   - Random rotations, shifts, and horizontal flips are applied to improve generalization.  
3. **CNN Architecture**:
   - Multiple convolutional layers to extract features like edges and textures.  
   - Pooling layers to downsample and reduce complexity.  
   - Dropout layers to prevent overfitting.  
4. **Training**:
   - The model is trained for 50 epochs with the Adam optimizer.  
5. **Evaluation**:
   - The model achieves ~80% accuracy on the test dataset.  


---

## **Tech Stack**
- **Programming Language**: Python  
- **Frameworks/Libraries**:  
  - TensorFlow  
  - Keras  
  - NumPy  
  - Matplotlib  

---

## **Dataset**
The **CIFAR-10 dataset** consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is split into:  
- **Training set**: 50,000 images  
- **Test set**: 10,000 images  

CIFAR-10 is included in TensorFlow/Keras and is loaded directly in the code.

---

## **Getting Started**
### Prerequisites
Make sure you have Python installed along with the required libraries. You can install the dependencies with:  

```bash
pip install tensorflow numpy matplotlib



Running the Code
1.Clone the repository:
git clone https://github.com/yourusername/CIFAR10-Image-Classifier.git
cd CIFAR10-Image-Classifier

2.Run the Python script:
python cifar10_image_classifier.py

3.The script will:
Preprocess and augment the CIFAR-10 dataset.
Train the CNN model for 50 epochs.
Save the trained model as cifar10_cnn_model.h5.


