#include "kf_eigen3.h"
#include "kf_math.h"

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
                                                         const Eigen::MatrixXd& in_process_noise,
                                                         const Eigen::MatrixXd& in_transition_process_noise_model,
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
    const Eigen::MatrixXd& Q = in_process_noise;
    const Eigen::MatrixXd& G = in_transition_process_noise_model;
    const Eigen::MatrixXd& H = in_transition_measurement_model;
          Eigen::MatrixXd& xp = state_predict;
          Eigen::MatrixXd& Pp = covariance_predict;
          Eigen::MatrixXd& zp = this->measurement_predict;


    MathematicsCommon::kf_predict<Eigen::MatrixXd>(xp,A,x,B,u,Pp,P,G,Q,zp,H);

    state = state_predict;
    covariance = covariance_predict;

    return std::make_pair(state,covariance);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> KFE::predict(const Eigen::MatrixXd& in_transition_state_model,
                                                         const Eigen::MatrixXd& in_transition_process_noise_model,
                                                         const Eigen::MatrixXd& in_transition_measurement_model)
{

    Eigen::MatrixXd state_predict;
    Eigen::MatrixXd covariance_predict;

    Eigen::MatrixXd control_model(this->transition_state_model.rows(),this->transition_state_model.cols());
    control_model.setZero();
    Eigen::MatrixXd control_input(this->state.rows(),this->state.cols());
    control_input.setZero();

    const Eigen::MatrixXd& x = this->state;
    const Eigen::MatrixXd& A = in_transition_state_model;
    const Eigen::MatrixXd& B = control_model;
    const Eigen::MatrixXd& u = control_input;
    const Eigen::MatrixXd& P = this->covariance;
    const Eigen::MatrixXd& Q = this->process_noise;
    const Eigen::MatrixXd& G = in_transition_process_noise_model;
    const Eigen::MatrixXd& H = in_transition_measurement_model;
          Eigen::MatrixXd& xp = state_predict;
          Eigen::MatrixXd& Pp = covariance_predict;
          Eigen::MatrixXd& zp = this->measurement_predict;

    MathematicsCommon::kf_predict<Eigen::MatrixXd>(xp,A,x,B,u,Pp,P,G,Q,zp,H);

    state = state_predict;
    covariance = covariance_predict;

    return std::make_pair(state,covariance);
}

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


    MathematicsCommon::kf_predict<Eigen::MatrixXd>(xp,A,x,B,u,Pp,P,G,Q,zp,H);

    state = state_predict;
    covariance = covariance_predict;

    return std::make_pair(state,covariance);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> KFE::predict()
{
    Eigen::MatrixXd state_predict;
    Eigen::MatrixXd covariance_predict;

    Eigen::MatrixXd control_model(this->transition_state_model.rows(),this->transition_state_model.cols());
    control_model.setZero();
    Eigen::MatrixXd control_input(this->state.rows(),this->state.cols());
    control_input.setZero();

    const Eigen::MatrixXd& x = this->state;
    const Eigen::MatrixXd& A = this->transition_state_model;
    const Eigen::MatrixXd& B = control_model;
    const Eigen::MatrixXd& u = control_input;
    const Eigen::MatrixXd& P = this->covariance;
    const Eigen::MatrixXd& Q = this->process_noise;
    const Eigen::MatrixXd& G = this->transition_process_noise_model;
    const Eigen::MatrixXd& H = this->transition_measurement_model;
          Eigen::MatrixXd& xp = state_predict;
          Eigen::MatrixXd& Pp = covariance_predict;
          Eigen::MatrixXd& zp = this->measurement_predict;


    MathematicsCommon::kf_predict<Eigen::MatrixXd>(xp,A,x,B,u,Pp,P,G,Q,zp,H);

    state = state_predict;
    covariance = covariance_predict;

    return std::make_pair(state,covariance);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> KFE::correct(const Eigen::MatrixXd& in_transition_measurement_model,
                                                         const Eigen::MatrixXd& in_measurement,
                                                         const Eigen::MatrixXd& in_measurement_noise)
{
    Eigen::MatrixXd state_correct;
    Eigen::MatrixXd covariance_correct;;

    const Eigen::MatrixXd& Pp = this->covariance;
    const Eigen::MatrixXd& H = in_transition_measurement_model;
    const Eigen::MatrixXd& R = in_measurement_noise;
    const Eigen::MatrixXd& xp = this->state;
    const Eigen::MatrixXd& zp = this->measurement_predict;
    const Eigen::MatrixXd& z = in_measurement;
          Eigen::MatrixXd& xc = state_correct;
          Eigen::MatrixXd& Pc = covariance_correct;

    MathematicsCommon::kf_correct<Eigen::MatrixXd>(H,Pp,R,xc,xp,z,zp,Pc);

    state = state_correct;
    covariance = covariance_correct;

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

    MathematicsCommon::kf_correct<Eigen::MatrixXd>(H,Pp,R,xc,xp,z,zp,Pc);

    state = state_correct;
    covariance = covariance_correct;

    return std::make_pair(state,covariance);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> KFE::correct(const Eigen::MatrixXd& in_measurement)
{
    Eigen::MatrixXd state_correct;
    Eigen::MatrixXd covariance_correct;;

    const Eigen::MatrixXd& Pp = this->covariance;
    const Eigen::MatrixXd& H = this->transition_measurement_model;
    const Eigen::MatrixXd& R = this->measurement_noise;
    const Eigen::MatrixXd& xp = this->state;
    const Eigen::MatrixXd& zp = this->measurement_predict;
    const Eigen::MatrixXd& z = in_measurement;
          Eigen::MatrixXd& xc = state_correct;
          Eigen::MatrixXd& Pc = covariance_correct;

    MathematicsCommon::kf_correct<Eigen::MatrixXd>(H,Pp,R,xc,xp,z,zp,Pc);

    state = state_correct;
    covariance = covariance_correct;

    return std::make_pair(state,covariance);
}



