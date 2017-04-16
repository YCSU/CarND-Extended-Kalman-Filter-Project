#include "kalman_filter.h"
#include <iostream>
using namespace std;
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
  P_ = ( MatrixXd::Identity(P_.rows(), P_.rows()) - K * H_ ) * P_;

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // State parameters
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  // H(x)
  float c1 = sqrt(px*px + py*py);
  if( c1 < 0.00001){
    return;
  }
  
  VectorXd H_of_x(3);
  H_of_x <<  c1, 
             atan2(py, px),
             (px*vx + py*vy) / c1;
  cout << "Hx" << H_of_x << endl;;

  // Calculate Jacobian 
  Tools cal_Hj;
  MatrixXd Hj = cal_Hj.CalculateJacobian(x_);

  VectorXd y = z - H_of_x;
  y(2) = fmod(y(2), (2 * M_PI));
  MatrixXd S = Hj * P_ * Hj.transpose() + R_;
  MatrixXd K = P_ * Hj.transpose() * S.inverse();

  x_ = x_ + K * y;
  P_ = ( MatrixXd::Identity(P_.rows(), P_.rows()) - K * Hj ) * P_;

}
