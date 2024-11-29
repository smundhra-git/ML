#ifndef SKLEARN_KMEANS_H
#define SKLEARN_KMEANS_H

#include <Eigen/Dense>

class KMeans {
public:
    KMeans() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_KMEANS_H

