#ifndef LIBRARYNAME_ONECLASSSVM_H
#define LIBRARYNAME_ONECLASSSVM_H

#include <Eigen/Dense>

class OneClassSVM {
public:
    OneClassSVM() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_ONECLASSSVM_H

