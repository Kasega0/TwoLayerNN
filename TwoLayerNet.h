//
// Created by Kasega0 on 2018/09/07.
//

#ifndef TWOLAYERNET_TWOLAYERNET_H
#define TWOLAYERNET_TWOLAYERNET_H

#include <map>
#include <Eigen/Dense>

class TwoLayerNet {
private:
    std::map<std::string, Eigen::MatrixXd> mParams;

private:
    Eigen::MatrixXd Sigmoid(Eigen::MatrixXd x);
    Eigen::MatrixXd Softmax(Eigen::MatrixXd x);
    double CrossEntropyError(Eigen::MatrixXd x, Eigen::VectorXd t);
    Eigen::MatrixXd NumericalGradientSub(std::function<double(Eigen::MatrixXd, Eigen::VectorXd)> f, Eigen::MatrixXd x, Eigen::VectorXd t);

public:
    TwoLayerNet(int input_size, int hidden_size, int output_size, double weight_init_std);
    Eigen::MatrixXd Predict(Eigen::MatrixXd x);
    double Loss(Eigen::MatrixXd x, Eigen::VectorXd t);
    double Accuracy(Eigen::MatrixXd x, Eigen::VectorXd t);
    std::map<std::string, Eigen::MatrixXd> NumericalGradient(Eigen::MatrixXd x, Eigen::VectorXd t);

};


#endif // TWOLAYERNET_TWOLAYERNET_H
