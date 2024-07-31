#ifndef LIBRARYNAME_FEWSHOTLEARNING_H
#define LIBRARYNAME_FEWSHOTLEARNING_H

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

#endif // LIBRARYNAME_FEWSHOTLEARNING_H

