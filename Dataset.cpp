#include <fstream>
#include "Dataset.h"
#include "directories.h"

using namespace std;

/** ビッグ・エンディアンのバイト列をintに変換する */
int big_endian(unsigned char* b) {
    int ret = 0;
    for (int i = 0; i < 4; i++) {
        ret = (ret << 8) | b[i];
    }
    return ret;
}

/** 画像データの読み取り
 *  @param[in] filename: 読み取るファイル名
 */
Eigen::MatrixXd load_data(const string& filename, bool normalize=true) {
    ifstream ifs(filename.c_str(), ios::in | ios::binary);
    if (!ifs.is_open()) {
        cerr << "Failed to open train data!!" << endl;
        exit(1);
    }

    // マジック・ナンバーを読む
    unsigned char b[4];
    ifs.read((char*)b, sizeof(char) * 4);

    // データの数を読む
    ifs.read((char*)b, sizeof(char) * 4);
    const int nimg = big_endian(b);

    // 画像の高さ(行数)を読む
    ifs.read((char*)b, sizeof(char) * 4);
    const int rows = big_endian(b);

    // 画像の幅(列数)を読む
    ifs.read((char*)b, sizeof(char) * 4);
    const int cols = big_endian(b);

    // 画像の画素データを読む
    unsigned char* buf = new unsigned char[rows * cols];
    Eigen::MatrixXd ret(rows * cols, nimg);
    const double div = normalize ? 255.0 : 1.0;
    for (int j = 0; j < nimg; j++) {
        ifs.read((char*)buf, sizeof(char) * rows * cols);
        for (int i = 0; i < rows * cols; i++) {
            ret(i, j) = buf[i] / div;
        }
    }
    delete[] buf;

    ifs.close();

    return move(ret);
}

/** ラベルの読み取り
 *  @param[in] filename: 読み取るファイル名
 */
Eigen::MatrixXd load_label(const string& filename) {
    ifstream ifs(filename.c_str(), ios::in | ios::binary);

    // マジック・ナンバーを読む
    unsigned char b[4];
    ifs.read((char*)b, sizeof(char) * 4);

    // データの数を読む
    ifs.read((char*)b, sizeof(char) * 4);
    int nimg = big_endian(b);

    // ラベルのデータを読む
    Eigen::MatrixXd ret(10, nimg);
    for (int i = 0; i < nimg; i++) {
        char digit;
        ifs.read((char*)&digit, sizeof(char));
        ret(digit, i) = 1.0;
    }

    ifs.close();

    return move(ret);
}

/** トレーニング用画像の読み取り */
Eigen::MatrixXd train_data(bool normalize=true) {
    return move(load_data(train_image_file, normalize));
}

/** トレーニング用ラベルの読み取り */
Eigen::MatrixXd train_label() {
    return move(load_label(train_label_file));
}

/** テスト用画像の読み取り */
Eigen::MatrixXd test_data(bool normalize=true) {
    return move(load_data(test_image_file, normalize));
}

/** テスト用ラベルの読み取り */
Eigen::MatrixXd test_label() {
    return move(load_label(test_label_file));
}