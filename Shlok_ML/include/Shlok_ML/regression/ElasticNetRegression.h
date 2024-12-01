#ifndef Shlok_ML_ELASTICNETREGRESSION_H
#define Shlok_ML_ELASTICNETREGRESSION_H

#include <Eigen/Dense>

class ElasticNetRegression {
private:
    Eigen::VectorXd coefficients;
    double l1_ratio;
    double alpha;

public:
    ElasticNetRegression(double l1_ratio = 0.5, double alpha = 1.0);

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, int max_iter = 1000, double tol = 1e-4);

    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;

    Eigen::VectorXd get_coefficients() const;

private:
    double soft_threshold(double rho, double lambda) const;
};

#endif // Shlok_ML_ELASTICNETREGRESSION_H
