# Shlok_ML - Custom Machine Learning Library

Shlok_ML is a custom machine learning library written in C++ with Python bindings using **pybind11**. It leverages **Eigen** for efficient numerical computations and supports various machine learning algorithms.

---

## Features

- **Regression Algorithms**:
  - ElasticNetRegression
  - LinearRegression
  - RidgeRegression
  - LassoRegression
  - PolynomialRegression

- **Classification Algorithms**:
  - SVM
  - Decision Trees
  - Random Forest
  - Naive Bayes

- **Clustering Algorithms**:
  - KMeans
  - DBSCAN
  - Gaussian Mixture Models (GMM)

- **Dimensionality Reduction**:
  - PCA
  - t-SNE
  - LDA
  - Autoencoders

---

## Requirements

- C++17 or higher
- Python 3.8 or higher
- **Dependencies**:
  - `pybind11` for Python bindings
  - `Eigen` for numerical computations

---

## Installation

### **1. Clone the Repository**

```bash
git clone https://github.com/smundhra/ML.git
cd Shlok_ML

### **2. Install Dependencies **
Install the required Python packages:

```bash

pip install -r requirements.txt
Ensure Eigen is installed on your system:

```bash

brew install eigen

### **3. Build the Project**
Create a build directory:

```bash

mkdir build
cd build
Configure the project with cmake:

```bash

cmake -DPython3_EXECUTABLE=$(which python3) -DEigen3_DIR=/opt/homebrew/opt/eigen/share/eigen3/cmake -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) ..

Compile the project:

```bash
make

### **4. Run Tests**
Use ctest to run all tests:

```bash

ctest

### **5. Use the Python Module**
You can now import the module in Python:

python

import Shlok_ML

# Example usage
from Shlok_ML import ElasticNetRegression

model = ElasticNetRegression(l1_ratio=0.5, alpha=1.0)
# Use `fit`, `predict`, and other methods
Directory Structure

Shlok_ML/
├── include/               # Header files for the algorithms
├── src/                   # Source files for the algorithms
├── tests/                 # Unit tests for each module
├── python/                # Python bindings for the library
├── build/                 # Build directory (generated)
├── CMakeLists.txt         # Build configuration file

