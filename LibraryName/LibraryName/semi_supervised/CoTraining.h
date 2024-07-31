#ifndef LIBRARYNAME_COTRAINING_H
#define LIBRARYNAME_COTRAINING_H

#include <Eigen/Dense>

class CoTraining {
public:
    CoTraining() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_COTRAINING_H

