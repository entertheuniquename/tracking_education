#pragma once

#include "utils.h"
#include "filter.h"

namespace Estimator
{
template <class M>
struct EKFMath
{
public:
    template<class SM, class JSM>
    void predict(M& xp,SM A,JSM JA, const M& x,const M& B,const M& u,
                    M& Pp,const M& P,const M& G,const M& Q,
                    double dt)
    {
        xp = A(x,dt) + B*u;
        Pp = JA(xp,dt)*P*Utils::transpose(JA(xp,dt)) + G*Q*Utils::transpose(G);
        Pp = (Pp + Utils::transpose(Pp))/2.;
    }

    template<class MM, class JMM>
    void correct(MM H,JMM JH, const M& Pp,const M& R,
                    M& xc,const M& xp,const M& z,
                    M& zp,M& Pc,M& Se, M& residue)
    {
          zp = H(xp);
          Se = JH()*Pp*Utils::transpose(JH()) + R;
        M K = Pp*Utils::transpose(JH())*Utils::inverse(Se);
          residue = z - zp;
          xc = xp + K * residue;
          Pc = Pp - K * Se * Utils::transpose(K);
          Pc = (Pc + Utils::transpose(Pc))/2.;
    }
};

template<class M, class SM, class MM, class GM, class JSM, class JMM>
class EKF : public EKFMath<M>, public Filter<M>
{
private:
    M state;
    M covariance;
    M process_noise;
    M measurement_noise;

    M state_predict;
    M covariance_predict;
    M measurement_predict;
    M covariance_of_measurement_predict;
    M residue;
public:
    EKF(M in_state,M in_covariance,M in_process_noise,M in_measurement_noise):
        state(in_state),
        covariance(in_covariance),
        process_noise(in_process_noise),
        measurement_noise(in_measurement_noise)
    {}

    EKF(const EKF& ekf):
        state(ekf.state),
        covariance(ekf.covariance),
        process_noise(ekf.process_noise),
        measurement_noise(ekf.measurement_noise),
        state_predict(ekf.state_predict),
        covariance_predict(ekf.covariance_predict),
        measurement_predict(ekf.measurement_predict),
        covariance_of_measurement_predict(ekf.covariance_of_measurement_predict),
        residue(ekf.residue)
    {}

    M getState()const override{return state;}
    M getCovariance()const override{return covariance;}
    M getStatePredict()const override{return state_predict;}
    M getCovariancePredict()const override{return covariance_predict;}
    M getMeasurementPredict()const override{return measurement_predict;}
    std::pair<M,M> getMeasurementPredictData(const double& dt)const override{return std::make_pair(M(),M());/*#ZAGL*/}
    M getCovarianceOfMeasurementPredict()const override{return covariance_of_measurement_predict;}
    bool setState(M& state_in)override{state = state_in;return true;}
    bool setCovariance(M& covariance_in)override{covariance = covariance_in;return true;}



    std::pair<M,M> predict(double dt)
    {
        M control_model = Utils::zero_matrix(this->state.rows(),this->state.rows());//#НЕ ПРОВЕРЕНО!
        M control_input = Utils::zero_matrix(this->state.rows(),this->state.cols());

        const M& x = this->state;
        const M& B = control_model;
        const M& u = control_input;
        const M& P = this->covariance;
        const M& Q = this->process_noise;
              M& xp = this->state_predict;
              M& Pp = this->covariance_predict;

        GM Gm;
        M G = Gm(dt);

        EKFMath<M>::predict(xp,SM(),JSM(),x,B,u,Pp,P,G,Q,dt);

        state = state_predict;
        covariance = covariance_predict;

        return std::make_pair(state,covariance);
    }

    std::pair<M,M> correct(const M& in_measurement)
    {
        M state_correct;
        M covariance_correct;;

        const M& Pp = this->covariance;
        const M& R = this->measurement_noise;
        const M& xp = this->state;
              M& zp = this->measurement_predict;
        const M& z = in_measurement;
              M& xc = state_correct;
              M& Pc = covariance_correct;
              M& Se = covariance_of_measurement_predict;
              M& dz = residue;

        EKFMath<M>::correct(MM(),JMM(),Pp,R,xc,xp,z,zp,Pc,Se,dz);

        state = state_correct;
        covariance = covariance_correct;

        return std::make_pair(state,covariance);
    }
};

}
