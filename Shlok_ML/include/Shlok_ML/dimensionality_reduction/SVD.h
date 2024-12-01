#ifndef Shlok_ML_SVD_H
#define Shlok_ML_SVD_H

#include <Eigen/Dense>

class SVD {
public:
    SVD() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // Shlok_ML_SVD_H

