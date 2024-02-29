#include "kf_eigen3.h"

KFE::KFE(Eigen::MatrixXd in_state,
       Eigen::MatrixXd in_covariance,
       Eigen::MatrixXd in_transition_state_model,
       Eigen::MatrixXd in_process_noise,
       Eigen::MatrixXd in_transition_process_noise_model,
       Eigen::MatrixXd in_transition_measurement_model,
       Eigen::MatrixXd in_measurement_noise):
    state(in_state),
    covariance(in_covariance),
    transition_state_model(in_transition_state_model),
    process_noise(in_process_noise),
    transition_process_noise_model(in_transition_process_noise_model),
    transition_measurement_model(in_transition_measurement_model),
    measurement_noise(in_measurement_noise)
{}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> KFE::predict(const Eigen::MatrixXd& in_transition_state_model,
                                                         const Eigen::MatrixXd& in_transition_measurement_model,
                                                         const Eigen::MatrixXd& in_control_input,
                                                         const Eigen::MatrixXd& in_control_model)
{
    Eigen::MatrixXd state_predict;
    Eigen::MatrixXd covariance_predict;

    const Eigen::MatrixXd& x = this->state;
    const Eigen::MatrixXd& A = in_transition_state_model;
    const Eigen::MatrixXd& B = in_control_model;
    const Eigen::MatrixXd& u = in_control_input;
    const Eigen::MatrixXd& P = this->covariance;
    const Eigen::MatrixXd& Q = this->process_noise;
    const Eigen::MatrixXd& G = this->transition_process_noise_model;
    const Eigen::MatrixXd& H = in_transition_measurement_model;
          Eigen::MatrixXd& xp = state_predict;
          Eigen::MatrixXd& Pp = covariance_predict;
          Eigen::MatrixXd& zp = this->measurement_predict;

    //std::cout << "predict:---------------" << std::endl;

    xp = A*x + B*u;
    Pp = A*P*A.transpose() + G*Q*G.transpose();
    Pp = (Pp + Pp.transpose())/2.;
    zp = H*x;

    //std::cout << "x:" << std::endl << x << std::endl;
    //std::cout << "xp:" << std::endl << xp << std::endl;
    //std::cout << "zp:" << std::endl << zp << std::endl;

    state = state_predict;
    covariance = covariance_predict;

    return std::make_pair(state,covariance);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> KFE::correct(const Eigen::MatrixXd& in_transition_measurement_model,
                                                         const Eigen::MatrixXd& in_measurement)
{
    Eigen::MatrixXd state_correct;
    Eigen::MatrixXd covariance_correct;;

    const Eigen::MatrixXd& Pp = this->covariance;
    const Eigen::MatrixXd& H = in_transition_measurement_model;
    const Eigen::MatrixXd& R = this->measurement_noise;
    const Eigen::MatrixXd& xp = this->state;
    const Eigen::MatrixXd& zp = this->measurement_predict;
    const Eigen::MatrixXd& z = in_measurement;
          Eigen::MatrixXd& xc = state_correct;
          Eigen::MatrixXd& Pc = covariance_correct;

    //std::cout << "correct:---------------" << std::endl;

    Eigen::MatrixXd S = H*Pp*H.transpose() + R;
    Eigen::MatrixXd K = Pp*H.transpose()*S.inverse();
                    xc = xp + K * (z - zp);
                    Pc = Pp - K * S * K.transpose();
                    Pc = (Pc + Pc.transpose())/2.;

    //std::cout << "xp:" << std::endl << xp << std::endl;
    //std::cout << "z:" << std::endl << z << std::endl;
    //std::cout << "zp:" << std::endl << zp << std::endl;
    //std::cout << "xc:" << std::endl << xc << std::endl;

    state = state_correct;
    covariance = covariance_correct;

    return std::make_pair(state,covariance);
}



