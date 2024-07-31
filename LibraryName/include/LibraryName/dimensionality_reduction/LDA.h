#ifndef LIBRARYNAME_LDA_H
#define LIBRARYNAME_LDA_H

#include <Eigen/Dense>

class LDA {
public:
    LDA() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_LDA_H

