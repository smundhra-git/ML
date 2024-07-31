#ifndef LIBRARYNAME_CNN_H
#define LIBRARYNAME_CNN_H

#include <Eigen/Dense>

class CNN {
public:
    CNN() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_CNN_H

