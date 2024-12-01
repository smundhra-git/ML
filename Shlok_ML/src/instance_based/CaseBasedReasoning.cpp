#include "Shlok_ML/instance_based/CaseBasedReasoning.h"

void CaseBasedReasoning::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd CaseBasedReasoning::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd CaseBasedReasoning::get_coefficients() const {
    return coefficients;
}

