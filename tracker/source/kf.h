#pragma once

#include "utils.h"
#include "filter.h"

namespace Estimator
{
template <class M>
struct KFMath
{
public:
    template<class SM, class GM>
    void predict(M& xp,SM A,const M& x,const M& B,const M& u,
                    M& Pp,const M& P,GM G,const M& Q,
                    double dt)
    {
        xp = A(x,dt,x) + B*u;
        Pp = Utils::transpose(A(Utils::transpose(A(P,dt,x)),dt,x)) + Utils::transpose(G(Utils::transpose(G(Q,dt)),dt));
        Pp = (Pp + Utils::transpose(Pp))/2.;
    }

    template<class MM>
    void correct(MM H,const M& Pp,const M& R,
                    M& xc,const M& xp,const M& z,
                    M& zp,M& Pc,M& Se, M& residue)
    {
          zp = H(xp,zp,xp);
          Se = Utils::transpose(H(Utils::transpose(H(Pp,zp,xp)),zp,xp)) + R;//#WARNING zp - проверить. //#SAVE //Utils::transpose(H(Utils::transpose(H(Pp)))) + R;
        M K = Utils::transpose(H(Utils::transpose(Pp),zp,xp))*Utils::inverse(Se);//#SAVE //Utils::transpose(H(Utils::transpose(Pp)))*Utils::inverse(Se);
          residue = z - zp;
          xc = xp + K * residue;
          Pc = Pp - K * Se * Utils::transpose(K);
          Pc = (Pc + Utils::transpose(Pc))/2.;
    }
};

template<class M, class SM, class MM, class GM>
class KF : public KFMath<M>, public Filter<M>
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
    KF(M in_state,M in_covariance,M in_process_noise,M in_measurement_noise):
        state(in_state),
        covariance(in_covariance),
        process_noise(in_process_noise),
        measurement_noise(in_measurement_noise)
    {
        //std::cout << "KF ========== ========== ========== ========== ==========" << std::endl;
        //std::cout << "state:" << std::endl << Utils::transpose(state) << std::endl;
        //std::cout << "covariance:" << std::endl << covariance << std::endl;
        //std::cout << "process_noise:" << std::endl << process_noise << std::endl;
        //std::cout << "measurement_noise:" << std::endl << measurement_noise << std::endl;
        //std::cout << "============= ========== ========== ========== ==========" << std::endl;
    }

    KF(const KF& kf):
        state(kf.state),
        covariance(kf.covariance),
        process_noise(kf.process_noise),
        measurement_noise(kf.measurement_noise),
        state_predict(kf.state_predict),
        covariance_predict(kf.covariance_predict),
        measurement_predict(kf.measurement_predict),
        covariance_of_measurement_predict(kf.covariance_of_measurement_predict),
        residue(kf.residue)
    {}

    M getProcessNoise()const override{return process_noise;}
    M getProcessNoise(double dt)const override{return Utils::transpose(GM()(Utils::transpose(GM()(process_noise,dt)),dt))/*GM()(dt)*process_noise*Utils::transpose(GM()(dt))*/;}
    M getMeasurementNoise()const override{return measurement_noise;}

    M getState()const override{return state;}
    M getCovariance()const override{return covariance;}
    M getStatePredict()const override{return state_predict;}
    M getCovariancePredict()const override{return covariance_predict;}
    M getMeasurementPredict()const override{return measurement_predict;}
    std::pair<M,M> getMeasurementPredictData(double dt)const override
    {
        M xp = SM()(state,dt,state);
        M Pp = Utils::transpose(SM()(Utils::transpose(SM()(covariance,dt,state)),dt,state)) + Utils::transpose(GM()(Utils::transpose(GM()(process_noise,dt)),dt))/*GM()(dt)*process_noise*Utils::transpose(GM()(dt))*/;
        Pp = (Pp + Utils::transpose(Pp))/2.;
        M zp = MM()(xp,zp,xp);
        M Se = Utils::transpose(MM()(Utils::transpose(MM()(Pp,zp,xp)),zp,xp)) + measurement_noise;
        return std::make_pair(zp,Se);
    }
    M getCovarianceOfMeasurementPredict()const override{return covariance_of_measurement_predict;}
    bool setState(M& state_in)override{state = state_in;return true;}
    bool setCovariance(M& covariance_in)override{covariance = covariance_in;return true;}

    std::pair<M,M> predict(double dt)
    {
        M control_model = Utils::zero_matrix(this->state.rows(),this->state.rows());//#TODO - НЕ ПРОВЕРЕНО!
        M control_input = Utils::zero_matrix(this->state.rows(),this->state.cols());

        const M& x = this->state;
        const M& B = control_model;
        const M& u = control_input;
        const M& P = this->covariance;
        const M& Q = this->process_noise;
              M& xp = this->state_predict;
              M& Pp = this->covariance_predict;

        KFMath<M>::predict(xp,SM(),x,B,u,Pp,P,GM(),Q,dt);

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

        KFMath<M>::correct(MM(),Pp,R,xc,xp,z,zp,Pc,Se,dz);

        state = state_correct;
        covariance = covariance_correct;

        return std::make_pair(state,covariance);
    }
};

}
