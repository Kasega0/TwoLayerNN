//
// Created by Kasega0 on 2018/09/07.
//
#include <random>
#include "TwoLayerNet.h"

using namespace std;
using namespace Eigen;

TwoLayerNet::TwoLayerNet(int input_size, int hidden_size, int output_size, double weight_init_std=0.01) {
    MatrixXd W1(input_size, hidden_size);
    MatrixXd b1 = MatrixXd::Zero(1, hidden_size);
    MatrixXd W2(hidden_size, output_size);
    MatrixXd b2 = MatrixXd::Zero(1, output_size);

    random_device seed;
    mt19937 engine(seed()); // メルセンヌ・ツイスター法
    normal_distribution<> dist(input_size, hidden_size);

    for (int i=0; i<W1.rows(); ++i) {
        for (int j=0; j<W1.cols(); ++j) {
            W1(i,j) = dist(engine);
        }
    }
    for (int i=0; i<W2.rows(); ++i) {
        for (int j=0; j<W2.cols(); ++j) {
            W2(i,j) = dist(engine);
        }
    }

    mParams["W1"] = W1;
    mParams["W2"] = W2;
    mParams["b1"] = b1;
    mParams["b2"] = b2;
}

MatrixXd TwoLayerNet::Predict(MatrixXd x){
    MatrixXd W1 = mParams["W1"], W2 = mParams["W2"];
    MatrixXd b1 = mParams["b1"], b2 = mParams["b2"];

    MatrixXd a1, z1, a2, y;
    a1 = x * W1 + b1;
    z1 = Sigmoid(a1);
    a2 = z1 * W2 + b2;
    y = Softmax(a2);

    return y;
}

double TwoLayerNet::Loss(MatrixXd x, VectorXd t){
    MatrixXd y = Predict(x);
    return CrossEntropyError(y, t);
}

double TwoLayerNet::Accuracy(MatrixXd x, VectorXd t){
    MatrixXd y = Predict(x);
    VectorXd::Index maxId[y.rows()];
    double accuracy_cnt = 0;
    for (int i=0; i<y.rows(); ++i) {
        y.row(i).maxCoeff(&maxId[i]);
        if(maxId[i] == t(i)) accuracy_cnt += 1.0;
    }

    return accuracy_cnt / y.rows();
}

map<string, MatrixXd> TwoLayerNet::NumericalGradient(MatrixXd x, VectorXd t){
    function<double(MatrixXd, VectorXd)> f = std::bind(&TwoLayerNet::Loss, this, placeholders::_1, placeholders::_2);

    map<string, MatrixXd> grads;
    grads["W1"] = NumericalGradientSub(f, mParams["W1"], t);
    grads["b1"] = NumericalGradientSub(f, mParams["b1"], t);
    grads["W2"] = NumericalGradientSub(f, mParams["W2"], t);
    grads["b2"] = NumericalGradientSub(f, mParams["b2"], t);

    return grads;
}

MatrixXd TwoLayerNet::NumericalGradientSub(function<double(MatrixXd, VectorXd)> f, MatrixXd x, VectorXd t){
    double h = 1e-4;
    MatrixXd grad = MatrixXd::Zero(x.rows(), x.cols());

    for (int i=0; i<x.rows(); ++i) {
        for (int j=0; j<x.cols(); ++j) {
            grad(i,j) = (f(x.array()+h,t) - f(x.array()-h,t)) / (2*h);
        }
    }

    return grad;
}

MatrixXd TwoLayerNet::Sigmoid(MatrixXd x){
    return (1 + (-1 * x).array().exp()).inverse();
}

MatrixXd TwoLayerNet::Softmax(MatrixXd x){
    double c = x.maxCoeff();
    x = (x.array() - c).exp();
    double s = x.sum();
    return x / s;
}

double TwoLayerNet::CrossEntropyError(MatrixXd x, VectorXd t) {
    double cee = 0;
    for (int i=0; i<x.rows(); ++i) {
        cee += log(x(i, t(i)) + 1e-7);
    }
    return -cee / x.rows();
}