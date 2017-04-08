#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_ * x_;
  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K = P_ * H_.transpose() * S.inverse();

  x_ = x_ + K * y;
  P_ = ( MatrixXd::Identity(P_.size(), P_.size()) - K * H_ ) * P_;

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // State parameters
  float px = z(0);
  float py = z(1);
  float vx = z(2);
  float vy = z(3);

  // H(z)
  float c1 = sqrt(px*px + py*py);
  VectorXd H_of_z;
  H_of_z <<  c1, 
             atan2(py, px),
             (px*vx + py*vy) / c1;

  // Calculate Jacobian 
  Tools cal_Hj;
  MatrixXd Hj = cal_Hj.CalculateJacobian(z);

  VectorXd y = z - H_of_z;
  MatrixXd S = Hj * P_ * Hj.transpose() + R_;
  MatrixXd K = P_ * Hj.transpose() * S.inverse();

  x_ = x_ + K * y;
  P_ = ( MatrixXd::Identity(P_.size(), P_.size()) - K * Hj ) * P_;
}
