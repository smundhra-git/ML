#ifndef LIBRARYNAME_MULTITASKLEARNING_H
#define LIBRARYNAME_MULTITASKLEARNING_H

#include <Eigen/Dense>

class MultiTaskLearning {
public:
    MultiTaskLearning() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_MULTITASKLEARNING_H

