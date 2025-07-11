# ML - Custom Machine Learning Library

**ML** is a custom Machine Learning library written primarily in **C++** (75%), with additional support from **Python** and **Fortran**. This library aims to provide modular, efficient, and extensible implementations of various machine learning algorithms, featuring Python bindings for broader usability. It is designed for performance and adaptability, leveraging **Eigen** for high-performance numerical computations.

---

## 📊 Technology Stack

- **Languages**: 
  - **C++** (75%)
  - **Fortran** (5%)
  - **CMake** (5%)
  - **Python** (4%)
  - **Others** (11%)

- **Libraries/Frameworks**:
  - **pybind11**: Python bindings for C++.
  - **Eigen**: High-performance numerical computation.
  - **CMake**: Build configuration and project management.

- **Testing Framework**:
  - **CTest** integrated with CMake.
  - **Pytest** for Python-level validations.

---

## 🚀 Current Status

The library currently includes the following implementations:

### **Available Modules**

- **Regression Algorithms**:
  - ElasticNetRegression
  - LinearRegression
  - RidgeRegression
  - LassoRegression
  - PolynomialRegression

- **Classification Algorithms**:
  - Support Vector Machines (SVM)
  - Decision Trees
  - Random Forest
  - Naive Bayes

- **Clustering Algorithms**:
  - KMeans
  - DBSCAN
  - Gaussian Mixture Models (GMM)

- **Dimensionality Reduction**:
  - PCA (Principal Component Analysis)
  - t-SNE
  - LDA (Linear Discriminant Analysis)
  - Autoencoders

- **Other Modules**:
  - **Anomaly Detection**:
    - Isolation Forest
    - Local Outlier Factor (LOF)
    - One-Class SVM
  - **Bayesian Learning**
  - **Ensemble Learning**
  - **Genetic Algorithms**
  - **Instance-Based Learning**
  - **Neural Networks**
  - **Reinforcement Learning**
  - **Semi-Supervised Learning**

---

## 🌟 Future Ideas

We are actively looking to expand the library with the following features:

- **Reinforcement Learning**:
  - Q-Learning
  - Deep Q-Networks (DQN)
  - Policy Gradient Methods

- **Ensemble Learning**:
  - Boosting
  - Bagging
  - Stacking

- **Neural Networks**:
  - Convolutional Neural Networks (CNNs)
  - Recurrent Neural Networks (RNNs)
  - Transformers

- **Bayesian Learning**:
  - Bayesian Optimization
  - Gaussian Naive Bayes
  - Bayesian Networks

- **Performance Enhancements**:
  - Parallelization of algorithms.
  - GPU-based computation for training.

## 💡 Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository and create your branch:
   ```bash
   git checkout -b feature/new_algorithm
Make your changes and add tests.

Push to your branch and create a pull request:

  ```bash
  git push origin feature/new_algorithm

```
Include a detailed explanation of your contribution.

📝 License
This project is licensed under the MIT License.

📞 Contact
If you have any questions or suggestions, feel free to reach out:

Author: Shlok Mundhra
Email: shlokmundhra@owu.edu
📂 Directory Structure

Shlok_ML/
├── include/               # Header files for the algorithms
├── src/                   # Source files for the algorithms
├── tests/                 # Unit tests for each module
├── python/                # Python bindings for the library
├── build/                 # Build directory (generated)
├── CMakeLists.txt         # Build configuration file
README.md              # Project documentation
