#ifndef SKLEARN_LSTM_H
#define SKLEARN_LSTM_H

#include <Eigen/Dense>

class LSTM {
public:
    LSTM() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_LSTM_H

