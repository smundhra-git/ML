#ifndef Shlok_ML_SVM_H
#define Shlok_ML_SVM_H

#include <Eigen/Dense>

class SVM {
public:
    SVM() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // Shlok_ML_SVM_H

