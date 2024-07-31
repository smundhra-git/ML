#include "LibraryName/anomaly_detection/AutoencoderAnomaly.h"

void AutoencoderAnomaly::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd AutoencoderAnomaly::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd AutoencoderAnomaly::get_coefficients() const {
    return coefficients;
}

