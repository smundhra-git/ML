#ifndef Shlok_ML_MULTITAShlok_MLING_H
#define Shlok_ML_MULTITAShlok_MLING_H

#include <Eigen/Dense>

class MultiTaShlok_MLing {
public:
    MultiTaShlok_MLing() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // Shlok_ML_MULTITAShlok_MLING_H

