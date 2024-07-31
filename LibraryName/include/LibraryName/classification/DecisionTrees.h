#ifndef LIBRARYNAME_DECISIONTREES_H
#define LIBRARYNAME_DECISIONTREES_H

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

#endif // LIBRARYNAME_DECISIONTREES_H

