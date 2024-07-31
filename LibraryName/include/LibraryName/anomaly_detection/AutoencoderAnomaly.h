#ifndef LIBRARYNAME_AUTOENCODERANOMALY_H
#define LIBRARYNAME_AUTOENCODERANOMALY_H

#include <Eigen/Dense>

class AutoencoderAnomaly {
public:
    AutoencoderAnomaly() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_AUTOENCODERANOMALY_H

