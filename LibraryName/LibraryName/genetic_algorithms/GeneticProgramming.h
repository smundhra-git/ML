#ifndef LIBRARYNAME_GENETICPROGRAMMING_H
#define LIBRARYNAME_GENETICPROGRAMMING_H

#include <Eigen/Dense>

class GeneticProgramming {
public:
    GeneticProgramming() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_GENETICPROGRAMMING_H

