#ifndef LIBRARYNAME_HIERARCHICALCLUSTERING_H
#define LIBRARYNAME_HIERARCHICALCLUSTERING_H

#include <Eigen/Dense>

class HierarchicalClustering {
public:
    HierarchicalClustering() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_HIERARCHICALCLUSTERING_H

