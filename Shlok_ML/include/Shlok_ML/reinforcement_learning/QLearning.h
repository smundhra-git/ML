#ifndef Shlok_ML_QLEARNING_H
#define Shlok_ML_QLEARNING_H

#include <Eigen/Dense>

class QLearning {
public:
    QLearning() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // Shlok_ML_QLEARNING_H

