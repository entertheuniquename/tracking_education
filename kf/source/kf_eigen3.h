#pragma once

#include "estimator.h"
#include<iostream>
#include<Eigen/Dense>

class KFE : public Estimator<Eigen::MatrixXd>
{
private:

    Eigen::MatrixXd state;
    Eigen::MatrixXd covariance;
    Eigen::MatrixXd transition_state_model;
    Eigen::MatrixXd process_noise;
    Eigen::MatrixXd transition_process_noise_model;
    Eigen::MatrixXd transition_measurement_model;
    Eigen::MatrixXd measurement_noise;
    Eigen::MatrixXd measurement_predict;

public:
    KFE(Eigen::MatrixXd in_state,
       Eigen::MatrixXd in_covariance,
       Eigen::MatrixXd in_transition_state_model,
       Eigen::MatrixXd in_process_noise,
       Eigen::MatrixXd in_transition_process_noise_model,
       Eigen::MatrixXd in_transition_measurement_model,
       Eigen::MatrixXd in_measurement_noise);
    std::pair<Eigen::MatrixXd , Eigen::MatrixXd > predict(const Eigen::MatrixXd& in_transition_state_model,
                                                          const Eigen::MatrixXd& in_transition_measurement_model,
                                                          const Eigen::MatrixXd& in_control_input,
                                                          const Eigen::MatrixXd& in_control_model);
    std::pair<Eigen::MatrixXd , Eigen::MatrixXd > correct(const Eigen::MatrixXd& in_transition_measurement_model,
                                                          const Eigen::MatrixXd& in_measurement);

    Eigen::MatrixXd get_state(){return state;}
    Eigen::MatrixXd get_covariance(){return covariance;}
};
