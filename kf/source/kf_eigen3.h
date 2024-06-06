#pragma once

#include "utils.h"

#include<iostream>
#include <Eigen/Dense>
namespace Estimator
{
template <class M = Eigen::MatrixXd>
struct KalmanFilterMath
{
public:
    void kf_predict(M& xp,const M& A,const M& x,const M& B,const M& u,
                    M& Pp,const M& P,const M& G,const M& Q,
                    M& zp,const M& H)
    {
        xp = A*x + B*u;
        Pp = A*P*Utils::transpose(A) + G*Q*Utils::transpose(G);
        Pp = (Pp + Utils::transpose(Pp))/2.;
        zp = H*xp;
    }

    void kf_correct(const M& H,const M& Pp,const M& R,
                    M& xc,const M& xp,const M& z,const M& zp,
                    M& Pc)
    {
        M S = H*Pp*Utils::transpose(H) + R;
        M K = Pp*Utils::transpose(H)*Utils::inverse(S);
          xc = xp + K * (z - zp);
          Pc = Pp - K * S * Utils::transpose(K);
          Pc = (Pc + Utils::transpose(Pc))/2.;
    }
};

template<class M, class SM, class MM>
class KFE : public KalmanFilterMath<M>
{
private:
    Eigen::MatrixXd state;//x0
    Eigen::MatrixXd covariance;//P0
    Eigen::MatrixXd transition_state_model;//A
    Eigen::MatrixXd process_noise;//Q
    Eigen::MatrixXd transition_process_noise_model;//G
    Eigen::MatrixXd transition_measurement_model;//M
    Eigen::MatrixXd measurement_noise;//R
    Eigen::MatrixXd measurement_predict;//zp
public:
    KFE(Eigen::MatrixXd in_state,
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

    Eigen::MatrixXd get_state(){return state;}
    Eigen::MatrixXd get_covariance(){return covariance;}

    std::pair<Eigen::MatrixXd,Eigen::MatrixXd> predict(double dt)
    {
        Eigen::MatrixXd state_predict;
        Eigen::MatrixXd covariance_predict;

        Eigen::MatrixXd control_model(this->transition_state_model.rows(),this->transition_state_model.cols());
        control_model.setZero();
        Eigen::MatrixXd control_input(this->state.rows(),this->state.cols());
        control_input.setZero();

        SM sm;
        MM mm;

        const Eigen::MatrixXd& x = this->state;
        const Eigen::MatrixXd& A = sm(dt);
        const Eigen::MatrixXd& B = control_model;
        const Eigen::MatrixXd& u = control_input;
        const Eigen::MatrixXd& P = this->covariance;
        const Eigen::MatrixXd& Q = this->process_noise;
        const Eigen::MatrixXd& G = this->transition_process_noise_model;
        const Eigen::MatrixXd& H = mm();
              Eigen::MatrixXd& xp = state_predict;
              Eigen::MatrixXd& Pp = covariance_predict;
              Eigen::MatrixXd& zp = this->measurement_predict;

        KalmanFilterMath<M>::kf_predict(xp,A,x,B,u,Pp,P,G,Q,zp,H);

        state = state_predict;
        covariance = covariance_predict;

        return std::make_pair(state,covariance);
    }
    std::pair<Eigen::MatrixXd,Eigen::MatrixXd> correct(const Eigen::MatrixXd& in_measurement)
    {
        Eigen::MatrixXd state_correct;
        Eigen::MatrixXd covariance_correct;;

        MM mm;

        const Eigen::MatrixXd& Pp = this->covariance;
        const Eigen::MatrixXd& H = mm();
        const Eigen::MatrixXd& R = this->measurement_noise;
        const Eigen::MatrixXd& xp = this->state;
        const Eigen::MatrixXd& zp = this->measurement_predict;
        const Eigen::MatrixXd& z = in_measurement;
              Eigen::MatrixXd& xc = state_correct;
              Eigen::MatrixXd& Pc = covariance_correct;

        KalmanFilterMath<M>::kf_correct(H,Pp,R,xc,xp,z,zp,Pc);

        state = state_correct;
        covariance = covariance_correct;

        return std::make_pair(state,covariance);
    }
};

//

template <class M = Eigen::MatrixXd>
struct KalmanFilterMath_X
{
public:
    template<class SMx>
    void kf_predict(M& xp,SMx A,const M& x,const M& B,const M& u,
                    M& Pp,const M& P,const M& G,const M& Q,
                    double dt)
    {
        xp = A(x,dt) + B*u;
        Pp = Utils::transpose(A(Utils::transpose(A(P,dt)),dt)) + G*Q*Utils::transpose(G);
        Pp = (Pp + Utils::transpose(Pp))/2.;
    }

    template<class MMx>
    void kf_correct(MMx H,const M& Pp,const M& R,
                    M& xc,const M& xp,const M& z,
                    M& zp, M& Pc)
    {
          zp = H(xp);
        M S = Utils::transpose(H(Utils::transpose(H(Pp)))) + R;
        M K = Utils::transpose(H(Utils::transpose(Pp)))*Utils::inverse(S);
          xc = xp + K * (z - zp);
          Pc = Pp - K * S * Utils::transpose(K);
          Pc = (Pc + Utils::transpose(Pc))/2.;
    }
};

template<class M, class SM, class MM>
class KFEx : public KalmanFilterMath_X<M>
{
private:
    Eigen::MatrixXd state;//x0
    Eigen::MatrixXd covariance;//P0
                 SM transition_state_model;//A
    Eigen::MatrixXd process_noise;//Q
    Eigen::MatrixXd transition_process_noise_model;//G
                 MM transition_measurement_model;//M
    Eigen::MatrixXd measurement_noise;//R
    Eigen::MatrixXd measurement_predict;//zp
public:
    KFEx(Eigen::MatrixXd in_state,
         Eigen::MatrixXd in_covariance,
                      SM in_transition_state_model,
         Eigen::MatrixXd in_process_noise,
         Eigen::MatrixXd in_transition_process_noise_model,
                      MM in_transition_measurement_model,
         Eigen::MatrixXd in_measurement_noise):
        state(in_state),
        covariance(in_covariance),
        transition_state_model(in_transition_state_model),
        process_noise(in_process_noise),
        transition_process_noise_model(in_transition_process_noise_model),
        transition_measurement_model(in_transition_measurement_model),
        measurement_noise(in_measurement_noise)
    {}

    Eigen::MatrixXd get_state(){return state;}
    Eigen::MatrixXd get_covariance(){return covariance;}

    std::pair<Eigen::MatrixXd,Eigen::MatrixXd> predict(double dt)
    {
        Eigen::MatrixXd state_predict;
        Eigen::MatrixXd covariance_predict;

        Eigen::MatrixXd control_model(7,7);//#TEMP - 7,7
        control_model.setZero();
        Eigen::MatrixXd control_input(this->state.rows(),this->state.cols());
        control_input.setZero();

        //SM sm;
        //MM mm;

        const Eigen::MatrixXd& x = this->state;
                            SM A = transition_state_model;
        const Eigen::MatrixXd& B = control_model;
        const Eigen::MatrixXd& u = control_input;
        const Eigen::MatrixXd& P = this->covariance;
        const Eigen::MatrixXd& Q = this->process_noise;
        const Eigen::MatrixXd& G = this->transition_process_noise_model;
        //const Eigen::MatrixXd& H = mm();
              Eigen::MatrixXd& xp = state_predict;
              Eigen::MatrixXd& Pp = covariance_predict;
              Eigen::MatrixXd& zp = this->measurement_predict;

        KalmanFilterMath_X<M>::kf_predict(xp,A,x,B,u,Pp,P,G,Q/*,zp,H*/,dt);

        state = state_predict;
        covariance = covariance_predict;

        return std::make_pair(state,covariance);
    }

    std::pair<Eigen::MatrixXd,Eigen::MatrixXd> correct(const Eigen::MatrixXd& in_measurement)
    {
        Eigen::MatrixXd state_correct;
        Eigen::MatrixXd covariance_correct;;

        const Eigen::MatrixXd& Pp = this->covariance;
                            MM H = transition_measurement_model;
        const Eigen::MatrixXd& R = this->measurement_noise;
        const Eigen::MatrixXd& xp = this->state;
              Eigen::MatrixXd& zp = this->measurement_predict;
        const Eigen::MatrixXd& z = in_measurement;
              Eigen::MatrixXd& xc = state_correct;
              Eigen::MatrixXd& Pc = covariance_correct;

        KalmanFilterMath_X<M>::kf_correct(H,Pp,R,xc,xp,z,zp,Pc);

        state = state_correct;
        covariance = covariance_correct;

        return std::make_pair(state,covariance);
    }
};

}
