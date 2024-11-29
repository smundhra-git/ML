#ifndef SKLEARN_FEWSHOTLEARNING_H
#define SKLEARN_FEWSHOTLEARNING_H

#include <Eigen/Dense>

class FewShotLearning {
public:
    FewShotLearning() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_FEWSHOTLEARNING_H

