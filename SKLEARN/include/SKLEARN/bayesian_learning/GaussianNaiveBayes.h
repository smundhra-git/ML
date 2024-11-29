#ifndef SKLEARN_GAUSSIANNAIVEBAYES_H
#define SKLEARN_GAUSSIANNAIVEBAYES_H

#include <Eigen/Dense>

class GaussianNaiveBayes {
public:
    GaussianNaiveBayes() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_GAUSSIANNAIVEBAYES_H

