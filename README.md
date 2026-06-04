# Telescope-Data-Classification
Telescope Data Classification: From Classical ML to Neural Network Tuning
You can run the code in Google colab, make sure to add data from left section. 
Or use your local jupyter notebook.

# Cherenkov Telescope Particle Classification
The link: https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
A comparative machine learning and deep learning approach designed to classify atmospheric telescope events into Gamma rays (signal) or Hadrons (background noise). This project documents the transition from classical linear baselines to non-linear geometric classifiers.

## Project Architecture & Methodology

The repository tracks a systematic machine learning workflow divided into four distinct phases:

1. **Data Preprocessing & Balance:** Handling data imbalances using random oversampling techniques and strict validation splitting to prevent data leakage.
2. **Linear Baselines:** Implementing Naive Bayes and Logistic Regression to establish a performance baseline and expose the limitations of linear decision boundaries on complex physics data.
3. **Non-Linear Geometric Classifiers:** Utilizing k-Nearest Neighbors (kNN) and Support Vector Machines (SVM) with Radial Basis Function (RBF) kernels to map high-dimensional, intertwined feature spaces.
4. **Deep Learning Optimization:** Constructing a multi-layer feedforward neural network utilizing an automated grid search pipeline to optimize internal architecture and regularizers.

---

## Technical Architecture (Neural Network)

The deep learning model is built using TensorFlow and the Keras Sequential API. It consists of a 3-layer architecture tailored for tabular physical measurements:

* **Input Layer:** Accepts 10 engineered physical features (e.g., length, width, size metrics).
* **Hidden Layer 1:** Fully connected layer (16, 32, or 64 nodes) with Rectified Linear Unit (ReLU) activation to introduce non-linearity.
* **Regularization (Dropout):** A dynamic dropout layer (up to 20%) to mitigate overfitting by randomly deactivating neurons during training steps.
* **Hidden Layer 2:** Second fully connected layer matching the node density of the first layer to map complex feature combinations.
* **Output Layer:** A single node wrapped in a Sigmoid activation function, mapping the internal structural signals to a clear probability distribution between 0.0 and 1.0.

---

## Hyperparameter Optimization Engine

To find the most stable and generalizable model, the pipeline runs an automated grid search across 54 unique parameter combinations:

* **Layer Density (Nodes):** `[16, 32, 64]`
* **Regularization (Dropout Probability):** `[0, 0.2]`
* **Learning Rates (Adam Optimizer):** `[0.01, 0.005, 0.001]`
* **Batch Sizes:** `[32, 64, 128]`

The selection logic evaluates each model dynamically using **Binary Crossentropy Loss** on an unseen validation dataset. By tracking confidence penalties rather than raw binary accuracy, the engine successfully filters out overfitted models that memorize training noise.

---

## Key Results & Model Evolution

| Model Architecture | Accuracy | Diagnostic Notes |
| :--- | :--- | :--- |
| **Naive Bayes** | 72% | Too simplistic; features violated independence assumptions. |
| **Logistic Regression** | 78% | Restricted by linear limitations; unable to separate dense clusters. |
| **k-Nearest Neighbors** | 82% | Improved performance via local distance clustering. |
| **SVM (RBF Kernel)** | 86% | Highly effective geometric boundary mapping. |
| **Optimized Deep Neural Network** | 87% (Train) / 80% (Valid) | High representation capacity; requires strict dropout to close the generalization gap. |

---

## Getting Started

### Prerequisites
* Python 3+
* TensorFlow 2.x
* Scikit-Learn
* Matplotlib
* NumPy
* Pandas
