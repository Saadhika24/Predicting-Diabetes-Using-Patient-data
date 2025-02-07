# Diabetic Retinopathy Prediction Using Multi-Classifier Approach

Diabetic Retinopathy (DR) is a serious eye disease caused by diabetes, which can lead to blindness if not diagnosed and treated early. This project utilizes a **multi-classifier approach** to predict the severity of DR using machine learning and deep learning techniques. By leveraging multiple classifiers, including **Convolutional Neural Networks (CNN), Support Vector Machines (SVM), Random Forest, k-Nearest Neighbors (k-NN), Gradient Boosting, and XGBoost**, the system improves accuracy and robustness in diagnosing DR from retinal fundus images.

## Dataset
- **Dataset Used**: Colored retinal images from `colored_images.zip`.
- **Data Type**: Fundus images labeled according to diabetic retinopathy severity.
- **Classification Categories**:
  1. No DR (Healthy)
  2. Mild DR
  3. Moderate DR
  4. Severe DR
  5. Proliferative DR
- **Annotations**: Each image is labeled according to severity to train the classifiers.

## Preprocessing Steps
- **Image Resizing**: Standardizing all images to a fixed size for input consistency.
- **Normalization**: Adjusting pixel intensity values for better model performance.
- **Data Augmentation**: Applying transformations such as rotation, flipping, and contrast adjustments.
- **Feature Extraction**: Extracting relevant patterns using **Histogram of Oriented Gradients (HOG), Local Binary Patterns (LBP), and Wavelet Transforms**.

## Models Used
This project uses a **multi-classifier approach** to compare and evaluate different models:
1. **Convolutional Neural Networks (CNN)** – Extracts spatial features from fundus images.
2. **Random Forest** – Tree-based ensemble model for classification.
3. **Support Vector Machine (SVM)** – Finds the optimal hyperplane for classification.
4. **k-Nearest Neighbors (k-NN)** – Classifies based on the closest labeled images.
5. **Gradient Boosting** – Sequential learning model to reduce classification errors.
6. **XGBoost** – Optimized gradient boosting algorithm for enhanced accuracy.

## Evaluation Metrics
The models are evaluated using:
- **Accuracy** – Measures the overall correctness of the predictions.
- **Precision, Recall, and F1-score** – Evaluates per-class classification performance.
- **AUC-ROC Curve** – Analyzes the performance of the classifiers.
- **Confusion Matrix** – Compares actual versus predicted classifications.

## Libraries & Tools Used
- **Google Colab** – Used for model training and execution.
- **Machine Learning & Deep Learning**: TensorFlow, PyTorch, Scikit-learn.
- **Computer Vision**: OpenCV.
- **Data Handling**: Pandas, NumPy.
- **Visualization**: Matplotlib, Seaborn.

## Installation Requirements
### **Prerequisites**
- **Google Colab Account** (No local installation required).
- Required dependencies installed using `pip` inside the Colab notebook.

### **Steps to Run in Google Colab**
1. Open Google Colab and upload the `.ipynb` notebook.
2. Upload `colored_images.zip` to Google Drive or Colab's working directory.
3. Mount Google Drive (if using Drive for storage):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Install dependencies:
   ```python
   !pip install tensorflow numpy pandas scikit-learn opencv-python matplotlib seaborn
   ```
5. Run the preprocessing and training cells.
6. Evaluate model performance using the provided metrics.
7. Upload new images and classify them using the trained model.

## Execution Steps
1. **Prepare the dataset**: Extract `colored_images.zip` and place it in the working directory.
2. **Run preprocessing**: Apply augmentation and feature extraction.
3. **Train the models**: Train CNN, SVM, k-NN, XGBoost, etc., on the dataset.
4. **Evaluate performance**: Compare classifiers using accuracy, F1-score, AUC-ROC.
5. **Make predictions**: Use trained models to classify new images.

Run the training script in Colab:
```python
!python train.py
```
Run the prediction script:
```python
!python predict.py --image path/to/image.jpg
```

This project aims to enhance diabetic retinopathy detection using AI-powered multi-classifiers, improving early diagnosis and patient care. 🚀

