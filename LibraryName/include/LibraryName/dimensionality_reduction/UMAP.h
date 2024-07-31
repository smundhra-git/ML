#ifndef LIBRARYNAME_UMAP_H
#define LIBRARYNAME_UMAP_H

#include <Eigen/Dense>

class UMAP {
public:
    UMAP() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_UMAP_H

