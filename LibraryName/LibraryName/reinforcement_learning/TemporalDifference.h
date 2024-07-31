#ifndef LIBRARYNAME_TEMPORALDIFFERENCE_H
#define LIBRARYNAME_TEMPORALDIFFERENCE_H

#include <Eigen/Dense>

class TemporalDifference {
public:
    TemporalDifference() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_TEMPORALDIFFERENCE_H

