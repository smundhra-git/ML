#include "SKLEARN/classification/SVM.h"

void SVM::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd SVM::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd SVM::get_coefficients() const {
    return coefficients;
}

