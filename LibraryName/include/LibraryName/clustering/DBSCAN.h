#ifndef LIBRARYNAME_DBSCAN_H
#define LIBRARYNAME_DBSCAN_H

#include <Eigen/Dense>

class DBSCAN {
public:
    DBSCAN() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_DBSCAN_H

