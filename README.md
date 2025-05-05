# 🧬 Breast Cancer Classification with Support Vector Machine (SVM)

This repository contains a machine learning project using **Support Vector Machine (SVM)** to classify whether a tumor is **benign** or **malignant** based on the **Breast Cancer** dataset. The notebook performs end-to-end preprocessing, training, and evaluation using Scikit-learn's SVM implementation.

---

## 📁 File Structure

- `Support_vector_machine.ipynb`  
  ➤ Jupyter Notebook containing all the code for loading, preprocessing, training, and evaluating the SVM classifier.

- `data/Breast_Cancer.csv`  
  ➤ Dataset used for classification (features include radius, texture, perimeter, area, smoothness, etc.).

---

## 🎯 Problem Statement

The objective is to create a machine learning model that can accurately predict whether a tumor is **malignant** or **benign** based on various diagnostic measurements. The classification is performed using a **Support Vector Machine**, known for its effectiveness in high-dimensional spaces.

---

## 🧠 Machine Learning Workflow

### 1. 🧼 Data Preprocessing

- Load `Breast_Cancer.csv` using `pandas`
- Inspect missing/null values and drop irrelevant columns (e.g., ID)
- Encode target labels (Benign = 0, Malignant = 1)
- Scale features using **StandardScaler**
- Split data into **train/test** sets (typically 80/20)

### 2. 🏗️ SVM Model Training

- Train SVM using `sklearn.svm.SVC`
- Try different kernels: `'linear'`, `'rbf'`, `'poly'`
- Use `C` and `gamma` parameters to control margin and fitting
- Optionally tune hyperparameters using **GridSearchCV**

### 3. 📊 Evaluation Metrics

- **Confusion Matrix**
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

### 4. 🔍 Visualization

- Confusion matrix heatmap with seaborn
- Decision boundaries (if reduced to 2D using PCA)
- Support vectors (for linear kernel)

---

## 🔧 Libraries Used

- **Python 3.x**
- **Pandas** – for data manipulation
- **NumPy** – for numerical operations
- **Matplotlib** & **Seaborn** – for visualization
- **Scikit-learn** – for model, scaling, splitting, and metrics

---

## 🚀 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/your-username/breast-cancer-svm.git
cd breast-cancer-svm
````

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Open the notebook

```bash
jupyter notebook Support_vector_machine.ipynb
```

---

## 💡 What You’ll Learn

* How to use SVM for medical diagnosis
* Difference between linear and non-linear kernels
* Effect of hyperparameters like `C` and `gamma`
* Importance of feature scaling in SVM
* Model evaluation with medical dataset metrics

---

## 👨‍💻 Author

Developed by **Nouhith**
Explore, fork, or contribute to improve the analysis and model accuracy!

----
