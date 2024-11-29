#include "SKLEARN/anomaly_detection/EllipticEnvelope.h"

void EllipticEnvelope::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd EllipticEnvelope::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd EllipticEnvelope::get_coefficients() const {
    return coefficients;
}

