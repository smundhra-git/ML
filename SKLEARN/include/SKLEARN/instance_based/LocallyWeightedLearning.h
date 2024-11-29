#ifndef SKLEARN_LOCALLYWEIGHTEDLEARNING_H
#define SKLEARN_LOCALLYWEIGHTEDLEARNING_H

#include <Eigen/Dense>

class LocallyWeightedLearning {
public:
    LocallyWeightedLearning() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_LOCALLYWEIGHTEDLEARNING_H

