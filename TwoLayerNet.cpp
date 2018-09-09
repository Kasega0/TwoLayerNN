//
// Created by Kasega0 on 2018/09/07.
//
#include <random>
#include <iostream>
#include "TwoLayerNet.h"

using namespace std;
using namespace Eigen;

TwoLayerNet::TwoLayerNet(int input_size, int hidden_size, int output_size, int batch_size, double weight_init_std) {
    MatrixXd W1(input_size, hidden_size);
    MatrixXd b1 = MatrixXd::Zero(batch_size, hidden_size);
    MatrixXd W2(hidden_size, output_size);
    MatrixXd b2 = MatrixXd::Zero(batch_size, output_size);

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

    /*cout << "mParams[\"W1\"] rows: " << W1.rows() << ", cols: " << W1.cols() << "\n";
    cout << "mParams[\"b1\"] rows: " << b1.rows() << ", cols: " << b1.cols() << "\n";
    cout << "mParams[\"W2\"] rows: " << W2.rows() << ", cols: " << W2.cols() << "\n";
    cout << "mParams[\"b2\"] rows: " << b2.rows() << ", cols: " << b2.cols() << "\n";*/
}

MatrixXd TwoLayerNet::Predict(MatrixXd x){
    MatrixXd W1 = mParams["W1"], W2 = mParams["W2"];
    MatrixXd b1 = mParams["b1"], b2 = mParams["b2"];
    /*cout << "Predict x rows: " << x.rows() << ", cols: " << x.cols() << "\n";
    cout << "Predict W1 rows: " << W1.rows() << ", cols: " << W1.cols() << "\n";
    cout << "Predict b1 rows: " << b1.rows() << ", cols: " << b1.cols() << "\n";*/

    MatrixXd a1, z1, a2, y;
    a1 = x * W1 + b1;
    z1 = Sigmoid(a1);
    a2 = z1 * W2 + b2;
    y = Softmax(a2);

    return y;
}

double TwoLayerNet::Loss(MatrixXd x, MatrixXd t){
    MatrixXd y = Predict(x);
    return CrossEntropyError(y, t);
}

double TwoLayerNet::Accuracy(MatrixXd x, MatrixXd t){
    MatrixXd y = Predict(x);
    MatrixXd::Index maxId[y.rows()];
    double accuracy_cnt = 0;
    for (int i=0; i<y.rows(); ++i) {
        y.row(i).maxCoeff(&maxId[i]);
        if(maxId[i] == t(i)) accuracy_cnt += 1.0;
    }

    return accuracy_cnt / y.rows();
}

map<string, MatrixXd> TwoLayerNet::NumericalGradient(MatrixXd x, MatrixXd t){
    //cout << "NumericalGradient(x) rows: " << x.rows() << ", cols: " << x.cols() << "\n";
    function<double(MatrixXd, MatrixXd)> f = std::bind(&TwoLayerNet::Loss, this, placeholders::_1, placeholders::_2);

    map<string, MatrixXd> grads;
    /*grads["W1"] = NumericalGradientSub(f, x, t);
    grads["b1"] = NumericalGradientSub(f, x, t);
    grads["W2"] = NumericalGradientSub(f, x, t);
    grads["b2"] = NumericalGradientSub(f, x, t);*/
    grads["W1"] = NumericalGradientSub2(f, x, t, "W1");
    grads["b1"] = NumericalGradientSub2(f, x, t, "b1");
    grads["W2"] = NumericalGradientSub2(f, x, t, "W2");
    grads["b2"] = NumericalGradientSub2(f, x, t, "b2");

    return grads;
}

MatrixXd TwoLayerNet::NumericalGradientSub(function<double(MatrixXd, MatrixXd)> f, MatrixXd x, MatrixXd t){
    //cout << "NumericalGradientSub(x) rows: " << x.rows() << ", cols: " << x.cols() << "\n";
    double h = 1e-4;
    MatrixXd grad = MatrixXd::Zero(x.rows(), x.cols());

    for (int i=0; i<x.rows(); ++i) {
        for (int j=0; j<x.cols(); ++j) {
            grad(i,j) = (f(x.array()+h,t) - f(x.array()-h,t)) / (2*h);
        }
    }

    return grad;
}

MatrixXd TwoLayerNet::NumericalGradientSub2(function<double(MatrixXd, MatrixXd)> f, MatrixXd x, MatrixXd t, string str_param){
    double h = 1e-4;
    MatrixXd prm = mParams[str_param];
    MatrixXd grad = MatrixXd::Zero(prm.rows(), prm.cols());

    for (int i=0; i<grad.rows(); ++i) {
        for (int j=0; j<grad.cols(); ++j) {
            mParams[str_param](i,j) += h;
            double f1 = f(x,t);
            mParams[str_param](i,j) -= 2*h;
            double f2 = f(x,t);
            grad(i,j) = (f1-f2) / (2*h);
            mParams[str_param](i,j) = prm(i,j);
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

double TwoLayerNet::CrossEntropyError(MatrixXd x, MatrixXd t) {
    /*cout << "CrossEntropyError x rows: " << x.rows() << ", cols: " << x.cols() << "\n";
    cout << "CrossEntropyError t rows: " << t.rows() << ", cols: " << t.cols() << "\n";*/
    x=((MatrixXd)(x.array() + 1e-7)).array().log();
    double tmp=0.0;
    for (int i=0; i<x.rows(); ++i) {
        tmp += x.row(i).dot(t.row(i));
    }
    return -tmp / x.rows();
    //return -(t * (MatrixXd)((MatrixXd)(x.array() + 1e-7)).array().log()).sum() / x.rows();
}