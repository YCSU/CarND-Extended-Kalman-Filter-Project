#include "kalman_filter.h"
#include <iostream>
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}


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
  
  // check division by zero
  float ro = sqrt(px*px + py*py);
  if( ro < 0.001){
    return;
  }

  // H(x)
  VectorXd H_of_x(3);
  H_of_x <<  ro, 
             atan2(py, px),
             (px*vx + py*vy) / ro;
   
  VectorXd y = z - H_of_x;
  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K = P_ * H_.transpose() * S.inverse();
  
  x_ = x_ + K * y;
  P_ = ( MatrixXd::Identity(x_.size(), x_.size()) - K * H_ ) * P_;

}
