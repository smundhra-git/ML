#include "LibraryName/other/FewShotLearning.h"

void FewShotLearning::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd FewShotLearning::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd FewShotLearning::get_coefficients() const {
    return coefficients;
}

