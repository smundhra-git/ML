#ifndef Shlok_ML_GANS_H
#define Shlok_ML_GANS_H

#include <Eigen/Dense>

class GANs {
public:
    GANs() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // Shlok_ML_GANS_H

