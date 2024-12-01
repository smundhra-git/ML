#include "Shlok_ML/classification/DecisionTrees.h"

void DecisionTrees::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd DecisionTrees::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd DecisionTrees::get_coefficients() const {
    return coefficients;
}

