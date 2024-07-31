#ifndef LIBRARYNAME_CASEBASEDREASONING_H
#define LIBRARYNAME_CASEBASEDREASONING_H

#include <Eigen/Dense>

class CaseBasedReasoning {
public:
    CaseBasedReasoning() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_CASEBASEDREASONING_H

