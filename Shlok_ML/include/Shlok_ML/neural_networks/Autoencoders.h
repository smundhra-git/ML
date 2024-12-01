#ifndef Shlok_ML_AUTOENCODERS_H
#define Shlok_ML_AUTOENCODERS_H

#include <Eigen/Dense>

class Autoencoders {
public:
    Autoencoders() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // Shlok_ML_AUTOENCODERS_H

