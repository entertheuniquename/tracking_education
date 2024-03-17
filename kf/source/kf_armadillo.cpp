#include "kf_armadillo.h"

KFA::KFA(arma::Mat<double> in_state,
       arma::Mat<double> in_covariance,
       arma::Mat<double> in_transition_state_model,
       arma::Mat<double> in_process_noise,
       arma::Mat<double> in_transition_process_noise,
       arma::Mat<double> in_transition_measurement_model,
       arma::Mat<double> in_measurement_noise):
    state(in_state),
    covariance(in_covariance),
    transition_state_model(in_transition_state_model),
    process_noise(in_process_noise),
    transition_process_noise(in_transition_process_noise),
    transition_measurement_model(in_transition_measurement_model),
    measurement_noise(in_measurement_noise)
{

}

std::pair<arma::Mat<double>, arma::Mat<double>> KFA::predict()
{
    return std::make_pair(arma::Mat<double>(1,1),arma::Mat<double>(1,1));//#ZAGL
}

std::pair<arma::Mat<double>, arma::Mat<double>> KFA::correct()
{
    return std::make_pair(arma::Mat<double>(1,1),arma::Mat<double>(1,1));//#ZAGL
}
