#ifndef LIBRARYNAME_BOOSTING_H
#define LIBRARYNAME_BOOSTING_H

#include <Eigen/Dense>

class Boosting {
public:
    Boosting() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_BOOSTING_H

