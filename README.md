# Virtual Reality Exposure Therapy with PPG Signal Classification

This repository contains code and data for a virtual reality exposure therapy (VRET) platform that assesses a user's mental state using photoplethysmography (PPG) signals. The project focuses on detecting relaxing and stressful states in subjects exposed to different VR environments.


## Introduction

Personalized virtual reality exposure therapy is a therapeutic practice that adapts to individual patients, leading to better health outcomes. Measuring a patient’s mental state to adjust the therapy is critical but challenging. Most studies use subjective methods to estimate a patient’s mental state, which can be inaccurate.

This project proposes a platform capable of assessing a patient’s mental state using non-intrusive and widely available physiological signals such as PPG. In a case study, we evaluate how PPG signals can be used to detect two states: relaxing and stressful.

## Data

The `Data` folder contains:

- **User Information**: Demographic data such as age and gender.
- **PPG Signal Data**: Subfolders like `PPG_W60000_O55000`, where:
  - `W60000` indicates a window size of 60,000 milliseconds (60 seconds).
  - `O55000` indicates an overlap of 55,000 milliseconds (55 seconds) between windows.

## Requirements

- Required Python libraries can be installed via `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Traditional Machine Learning Methods

The `classifiers.ipynb` notebook runs various traditional machine learning methods:

- Support Vector Machine (SVM)
- Random Forest
- AdaBoost with Decision Trees
- Neural Network (shallow)
- Linear Discriminant Analysis
- Stochastic Gradient Descent (SGD)
- K-Nearest Neighbors (KNN)

To run the notebook:

1. Open `classifiers.ipynb` in Jupyter Notebook or JupyterLab.
2. Run the cells sequentially to execute the code.

### Deep Learning Method

The `DeepLearning.ipynb` notebook implements a deep neural network for classification.

To run the notebook:

1. Open `DeepLearning.ipynb` in Jupyter Notebook or JupyterLab.
2. Ensure that PyTorch is installed and configured correctly.
3. Run the cells sequentially to execute the code.

## Methods

- **Data Collection**: 19 healthy subjects were exposed to two VR environments: relaxing and stressful.
- **Signal Processing**: PPG signals were segmented using different window sizes with overlap.
- **Feature Extraction**: Relevant features were extracted from the PPG signals.
- **Leave-One-Subject-Out (LOSO) Cross-Validation**: Used to evaluate the model's ability to generalize to unseen subjects.
- **Machine Learning Models**: Both traditional machine learning algorithms and deep learning models were employed.
- **Subject Information Embedding**: Incorporating demographic information (age, gender, video game and VR experience, and time since last stimulant intake) to improve model performance.

## Results

- Using LOSO cross-validation, our best classification model predicted relaxing and stressful states with approximately **62% accuracy**, outperforming many more complex approaches.
- By adding subject information (embedding), accuracy increased to **65%**.

## VR Environments

You can view examples of the VR environments used in this study:

- **Relaxing Environment**: [![Relaxing Environment](https://img.youtube.com/vi/iceV8TMDgZE/0.jpg)](https://youtu.be/iceV8TMDgZE)
- **Stressful Environment**: [![Stressful Environment](https://img.youtube.com/vi/2piO3lCVX50/0.jpg)](https://youtu.be/2piO3lCVX50)

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

---

*Note: This project is for research purposes. The results are based on a limited dataset of 19 subjects and may not generalize to a broader population.*
