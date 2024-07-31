#ifndef LIBRARYNAME_TSNE_H
#define LIBRARYNAME_TSNE_H

#include <Eigen/Dense>

class TSNE {
public:
    TSNE() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_TSNE_H

