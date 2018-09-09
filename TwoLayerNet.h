//
// Created by Kasega0 on 2018/09/07.
//

#ifndef TWOLAYERNET_TWOLAYERNET_H
#define TWOLAYERNET_TWOLAYERNET_H

#include <map>
#include <Eigen/Dense>

class TwoLayerNet {
public:
    std::map<std::string, Eigen::MatrixXd> mParams;

private:
    Eigen::MatrixXd Sigmoid(Eigen::MatrixXd x);
    Eigen::MatrixXd Softmax(Eigen::MatrixXd x);
    double CrossEntropyError(Eigen::MatrixXd x, Eigen::MatrixXd t);
    Eigen::MatrixXd NumericalGradientSub(std::function<double(Eigen::MatrixXd, Eigen::MatrixXd)> f, Eigen::MatrixXd x, Eigen::MatrixXd t);
    Eigen::MatrixXd NumericalGradientSub2(std::function<double(Eigen::MatrixXd, Eigen::MatrixXd)> f, Eigen::MatrixXd x, Eigen::MatrixXd t, std::string str_param);

public:
    TwoLayerNet(int input_size, int hidden_size, int output_size, int batch_size, double weight_init_std=0.01);
    Eigen::MatrixXd Predict(Eigen::MatrixXd x);
    double Loss(Eigen::MatrixXd x, Eigen::MatrixXd t);
    double Accuracy(Eigen::MatrixXd x, Eigen::MatrixXd t);
    std::map<std::string, Eigen::MatrixXd> NumericalGradient(Eigen::MatrixXd x, Eigen::MatrixXd t);

};


#endif // TWOLAYERNET_TWOLAYERNET_H
