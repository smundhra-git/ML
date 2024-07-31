#ifndef LIBRARYNAME_ELASTICNETREGRESSION_H
#define LIBRARYNAME_ELASTICNETREGRESSION_H

#include <Eigen/Dense>

class ElasticNetRegression {
public:
    ElasticNetRegression() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_ELASTICNETREGRESSION_H

