#include "SKLEARN/classification/RandomForest.h"
#include <iostream>
#include <Eigen/Dense>

using namespace std;

int main() {
    Eigen::MatrixXd X(100, 2);
    Eigen::VectorXd y(100);
    // Generate some example data
    srand((unsigned)time(0));
    for (int i = 0; i < 100; ++i) {
        X(i, 0) = rand() % 100 / 10.0;
        X(i, 1) = rand() % 100 / 10.0;
        y(i) = 3 * X(i, 0) - 2 * X(i, 1) + (rand() % 10 / 10.0);
    }

    RandomForest model;
    model.fit(X, y);

    Eigen::VectorXd predictions = model.predict(X);
    cout << "Coefficients: " << model.get_coefficients().transpose() << endl;
    cout << "First 10 Predictions: " << predictions.head(10).transpose() << endl;

    return 0;
}

