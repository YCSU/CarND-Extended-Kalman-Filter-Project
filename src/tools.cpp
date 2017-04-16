#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
  VectorXd rmse(4);
  rmse << 0., 0., 0., 0.;
  if( estimations.size() == 0 || estimations.size() != ground_truth.size() )
  {
  	std::cout << "estimation has some problems" << std::endl;
  	return rmse;
  }
  
  // squared sum
  VectorXd diff;
  for( int i = 0; i < estimations.size(); i++)
  {
	  diff = estimations[i] - ground_truth[i];
    diff = diff.array() * diff.array();
	  rmse += diff;
  }

  // take mean
  rmse /= rmse.size();

  // take root
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) 
{
  
  MatrixXd Hj(3,4);
  // state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // some parameters calculation
  float c1 = px*px + py*py;
  float c2 = sqrt(c1);
  float c3 = c1*c2;

  // check division by zero
  if( c1 < 0.001)
  {
  	std::cout << "Jacobian --- Division by Zero" << std::endl;
  	return Hj; 
  }
  
  // calculate Jacobian 
  Hj << px/c2, py/c2, 0, 0,
  		-py/c1, px/c1, 0, 0,
        py*(vx*py-vy*px)/c3, px*(vy*px-vx*py)/c3, px/c2, py/c2;

  return Hj;
}
