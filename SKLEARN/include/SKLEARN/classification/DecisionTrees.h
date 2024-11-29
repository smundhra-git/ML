#ifndef SKLEARN_DECISIONTREES_H
#define SKLEARN_DECISIONTREES_H

#include <Eigen/Dense>

class DecisionTrees {
public:
    DecisionTrees() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_DECISIONTREES_H

