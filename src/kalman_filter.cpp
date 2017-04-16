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
  //cout << "y: " << endl << y << endl;
  //cout << "S: " << endl << S << endl;
  //cout << "K: " << endl << K << endl;
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
  float ro = sqrt(px*px + py*py);
  if( ro < 0.03){
    return;
  }
  VectorXd H_of_x(3);
  H_of_x <<  ro, 
             atan2(py, px),
             (px*vx + py*vy) / ro;
   
  VectorXd y = z - H_of_x;
  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K = P_ * H_.transpose() * S.inverse();

  x_ = x_ + K * y;
  P_ = ( MatrixXd::Identity(P_.rows(), P_.rows()) - K * H_ ) * P_;

}
