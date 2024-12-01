#ifndef Shlok_ML_TSNE_H
#define Shlok_ML_TSNE_H

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

#endif // Shlok_ML_TSNE_H
