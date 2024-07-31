#ifndef LIBRARYNAME_KMEANS_H
#define LIBRARYNAME_KMEANS_H

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

#endif // LIBRARYNAME_KMEANS_H

