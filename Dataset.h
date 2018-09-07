#ifndef TWOLAYERNET_DATASET_H
#define TWOLAYERNET_DATASET_H

#include <iostream>
#include <Eigen/Dense>

/** トレーニング用の画像を読み込む */
Eigen::MatrixXd train_data(bool normalize=true);

/** トレーニング用のラベルを読み込む */
Eigen::MatrixXd train_label();

/** テスト用の画像を読み込む */
Eigen::MatrixXd test_data(bool normalize=true);

/** テスト用のラベルを読み込む */
Eigen::MatrixXd test_label();

#endif // TWOLAYERNET_DATASET_H
