#ifndef LIBRARYNAME_LOGISTICREGRESSION_H
#define LIBRARYNAME_LOGISTICREGRESSION_H

#include <Eigen/Dense>

class LogisticRegression {
public:
    LogisticRegression() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_LOGISTICREGRESSION_H

