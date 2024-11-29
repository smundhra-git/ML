#include "SKLEARN/regression/ElasticNetRegression.h"

ElasticNetRegression::ElasticNetRegression(double l1_ratio, double alpha)
    : l1_ratio(l1_ratio), alpha(alpha) {}

void ElasticNetRegression::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, int max_iter, double tol) {
    int m = X.rows();
    int n = X.cols();
    coefficients = Eigen::VectorXd::Zero(n);

    Eigen::VectorXd prev_coefficients = coefficients;
    for (int iter = 0; iter < max_iter; ++iter) {
        for (int j = 0; j < n; ++j) {
            double rho = (X.col(j).array() * (y - (X * coefficients)).array()).sum() + coefficients(j) * X.col(j).squaredNorm();
            double z = X.col(j).squaredNorm();
            coefficients(j) = soft_threshold(rho, alpha * l1_ratio) / (z + alpha * (1 - l1_ratio));
        }

        if ((coefficients - prev_coefficients).norm() < tol) {
            break;
        }
        prev_coefficients = coefficients;
    }
}

Eigen::VectorXd ElasticNetRegression::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd ElasticNetRegression::get_coefficients() const {
    return coefficients;
}

double ElasticNetRegression::soft_threshold(double rho, double lambda) const {
    if (rho < -lambda) {
        return rho + lambda;
    } else if (rho > lambda) {
        return rho - lambda;
    } else {
        return 0.0;
    }
}
