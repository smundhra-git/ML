#ifndef LIBRARYNAME_SELFTRAINING_H
#define LIBRARYNAME_SELFTRAINING_H

#include <Eigen/Dense>

class SelfTraining {
public:
    SelfTraining() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_SELFTRAINING_H

