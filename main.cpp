#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <cstdlib> // EXIT_FAILUREのため

#include "Dataset.h"
#include "directories.h"
#include "TwoLayerNet.h"

using namespace Eigen;
using namespace std;

void shuffle(int array[], int size) {
    int i = size;
    while (i > 1) {
        int j = rand() % i;
        i--;
        int t = array[i];
        array[i] = array[j];
        array[j] = t;
    }
}

int main() {
    const string prm_name[4] = { "W1", "b1", "W2", "b2", };

    MatrixXd x_train = train_data(true);
    MatrixXd t_train = train_label();

    //ハイパーパラメタ
    const int iters_num = 10000; //繰り返し回数
    const int train_size = x_train.rows();
    const int batch_size = 100; //一度のサンプル数
    const double learnign_rate = 0.1;

    int *values = new int[train_size];
    for (int i = 0; i < train_size; i++) { values[i]=i; }
    //for (int i = 0; i < batch_size; i++) { batch_mask[i]=i; }

    TwoLayerNet network = TwoLayerNet(x_train.cols(), 50, 10, batch_size, learnign_rate);

    MatrixXd x_batch= MatrixXd::Zero(batch_size, x_train.cols());
    MatrixXd t_batch= MatrixXd::Zero(batch_size, t_train.cols());
    /*cout << "Xb rows: " << x_batch.rows() << ", cols: " << x_batch.cols() << "\n";
    cout << "Tb rows: " << t_batch.rows() << ", cols: " << t_batch.cols() << "\n";*/

    double *train_loss_list = new double[iters_num];
    for(int k=0; k<iters_num; ++k){
        cout << k+1 << '/' << iters_num << "\n";
        //ミニバッチの取得
        shuffle(values, train_size);
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < x_train.cols(); ++j) {
                x_batch(i,j) = x_train(values[i],j);
            }
            for (int j = 0; j < t_train.cols(); ++j) {
                t_batch(i,j) = t_train(values[i],j);
            }
        }

        //勾配計算
        map<string, MatrixXd> grad = network.NumericalGradient(x_batch, t_batch);
        /*cout << "grad[\"W1\"] rows: " << grad["W1"].rows() << ", cols: " << grad["W1"].cols() << "\n";
        cout << "grad[\"b1\"] rows: " << grad["b1"].rows() << ", cols: " << grad["b1"].cols() << "\n";
        cout << "grad[\"W2\"] rows: " << grad["W2"].rows() << ", cols: " << grad["W2"].cols() << "\n";
        cout << "grad[\"b2\"] rows: " << grad["b2"].rows() << ", cols: " << grad["b2"].cols() << "\n";*/
        for(int i=0; i<4; ++i){
            network.mParams[prm_name[i]] -= (MatrixXd)(learnign_rate * grad[prm_name[i]].array());
        }
        /*network.mParams["W1"] -= (MatrixXd)(learnign_rate * grad["W1"].array());
        network.mParams["b1"] -= (MatrixXd)(learnign_rate * grad["b1"].array());
        network.mParams["W2"] -= (MatrixXd)(learnign_rate * grad["W2"].array());
        network.mParams["b2"] -= (MatrixXd)(learnign_rate * grad["b2"].array());*/

        double loss = network.Loss(x_batch, t_batch);
        train_loss_list[k] = loss;
    }

    cout << train_loss_list[0] << " " << flush;
    for(int i=1; i<iters_num; ++i){
        cout << train_loss_list[i] << " " << flush;
        if(i%30==0) cout << "\n";
    }

    const string filename = root_directory + "result_params.txt"; //directories.h
    fstream fs;
    fs.open(filename, ios::out);
    if(! fs.is_open()) {
        return EXIT_FAILURE;
    }
    for(int i=0; i<4; ++i){
        fs << "\"" << prm_name[i] << "\": " << network.mParams[prm_name[i]] << ',' << "\n";
    }
    fs.close();

    delete[] train_loss_list;
    delete[] values;
    return 0;
}