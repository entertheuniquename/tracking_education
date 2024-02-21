#pragma once

#include<armadillo>
//#include<eigen3/Eigen/Eigen>

class KFA
{
private:

    arma::Mat<double> state;
    arma::Mat<double> covariance;
    arma::Mat<double> transition_state_model;
    arma::Mat<double> process_noise;
    arma::Mat<double> transition_process_noise;
    arma::Mat<double> transition_measurement_model;
    arma::Mat<double> measurement_noise;

public:
    KFA(arma::Mat<double> in_state,
       arma::Mat<double> in_covariance,
       arma::Mat<double> in_transition_state_model,
       arma::Mat<double> in_process_noise,
       arma::Mat<double> in_transition_process_noise,
       arma::Mat<double> in_transition_measurement_model,
       arma::Mat<double> in_measurement_noise);
    std::pair<arma::Mat<double>, arma::Mat<double>> predict();
    std::pair<arma::Mat<double>, arma::Mat<double>> correct();
};
