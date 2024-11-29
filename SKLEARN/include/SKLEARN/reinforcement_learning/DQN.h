#ifndef SKLEARN_DQN_H
#define SKLEARN_DQN_H

#include <Eigen/Dense>

class DQN {
public:
    DQN() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_DQN_H

