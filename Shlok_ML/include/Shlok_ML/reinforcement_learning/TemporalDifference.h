#ifndef Shlok_ML_TEMPORALDIFFERENCE_H
#define Shlok_ML_TEMPORALDIFFERENCE_H

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

#endif // Shlok_ML_TEMPORALDIFFERENCE_H

