#pragma once

#include "kf.h"
#include "ekf_qr.h"
#include "models.h"
#include <math.h>

namespace Estimator
{
template<class M, class E1, class E2, class E3>
class IMM
{
private:
    M state;
    M covariance;
    M state_predict;
    M covariance_predict;
    E1 estimator1;
    E2 estimator2;
    E3 estimator3;
    M mu;
    M tp;
public:
    struct Mixes{M X01;M P01;M X02; M P02;M X03; M P03;};

    IMM(E1 e1, E2 e2, E2 e3, M m, M t):
        estimator1(e1),
        estimator2(e2),
        estimator3(e3),
        mu(m),
        tp(t){}

    IMM(M state,M covariance,M process_noise,M measureNoise, M m, M t):
        estimator1(state,covariance,process_noise,measureNoise),
        estimator2(state,covariance,process_noise,measureNoise),
        estimator3(state,covariance,process_noise,measureNoise),
        mu(m),
        tp(t){}

    M mixing_probability()
    {
        M mx(3,3);
        mx.setZero();
        M fi(1,3);
        fi.setZero();

        fi(0) = (mu(0)*tp(0,0)+mu(1)*tp(1,0)+mu(2)*tp(2,0));//#TODO - переделать на перемножение матриц!
        fi(1) = (mu(0)*tp(0,1)+mu(1)*tp(1,1)+mu(2)*tp(2,1));//#TODO - переделать на перемножение матриц!
        fi(2) = (mu(0)*tp(0,2)+mu(1)*tp(1,2)+mu(2)*tp(2,2));//#TODO - переделать на перемножение матриц!

        mx(0,0) = tp(0,0)*mu(0)/fi(0);//#TODO - переделать на матричные операции!
        mx(1,0) = tp(1,0)*mu(1)/fi(0);//#TODO - переделать на матричные операции!
        mx(2,0) = tp(2,0)*mu(2)/fi(0);//#TODO - переделать на матричные операции!

        mx(0,1) = tp(0,1)*mu(0)/fi(1);//#TODO - переделать на матричные операции!
        mx(1,1) = tp(1,1)*mu(1)/fi(1);//#TODO - переделать на матричные операции!
        mx(2,1) = tp(2,1)*mu(2)/fi(1);//#TODO - переделать на матричные операции!

        mx(0,2) = tp(0,2)*mu(0)/fi(2);//#TODO - переделать на матричные операции!
        mx(1,2) = tp(1,2)*mu(1)/fi(2);//#TODO - переделать на матричные операции!
        mx(2,2) = tp(2,2)*mu(2)/fi(2);//#TODO - переделать на матричные операции!

        return mx;
    }

    Mixes mixed_states_and_covariances()
    {
        M mx = mixing_probability();
        Mixes mxs;

        M X1 = estimator1.getState();
        M X2 = estimator2.getState();
        M X3 = estimator3.getState();
        M P1 = estimator1.getCovariance();
        M P2 = estimator2.getCovariance();
        M P3 = estimator3.getCovariance();

        mxs.X01 = mx(0,0)*X1+mx(1,0)*X2+mx(2,0)*X3;
        mxs.X02 = mx(0,1)*X1+mx(1,1)*X2+mx(2,1)*X3;
        mxs.X03 = mx(0,2)*X1+mx(1,2)*X2+mx(2,2)*X3;

        M dX101 = X1-mxs.X01;
        M dX201 = X2-mxs.X01;
        M dX301 = X3-mxs.X01;
        M dX102 = X1-mxs.X02;
        M dX202 = X2-mxs.X02;
        M dX302 = X3-mxs.X02;
        M dX103 = X1-mxs.X03;
        M dX203 = X2-mxs.X03;
        M dX303 = X3-mxs.X03;

        mxs.P01 = ( mx(0,0)*(P1+dX101*Utils::transpose(dX101)) +
                    mx(1,0)*(P2+dX201*Utils::transpose(dX201)) +
                    mx(2,0)*(P3+dX301*Utils::transpose(dX301))
                  );
        mxs.P02 = ( mx(0,1)*(P1+dX102*Utils::transpose(dX102)) +
                    mx(1,1)*(P2+dX202*Utils::transpose(dX202)) +
                    mx(2,1)*(P3+dX302*Utils::transpose(dX302))
                  );
        mxs.P03 = ( mx(0,2)*(P1+dX103*Utils::transpose(dX103)) +
                    mx(1,2)*(P2+dX203*Utils::transpose(dX203)) +
                    mx(2,2)*(P3+dX303*Utils::transpose(dX303))
                  );

        return mxs;
    }

    double system_probability(M& z,M& ze,M& Se)
    {
        double n = 2;//#TODO - imm - n-?
        M residual = z-ze;
        M power = -0.5*residual*Utils::inverse(Se)*Utils::transpose(residual);
        return std::pow((1./2./M_PI),(n/2))/std::sqrt(Se.determinant())*std::exp(power.determinant());//#TODO - сверить с формулой!;
    }

    M models_probability_recalculation(M& z, M& ze1, M& ze2, M& ze3, M& Se1, M& Se2, M& Se3)
    {
        M mx = mixing_probability();
        mu(0) = system_probability(z,ze1,Se1)*(mx(0,0)*mu(0)+mx(1,0)*mu(1)+mx(2,0)*mu(2));
        mu(1) = system_probability(z,ze2,Se2)*(mx(0,1)*mu(0)+mx(1,1)*mu(1)+mx(2,1)*mu(2));
        mu(2) = system_probability(z,ze3,Se3)*(mx(0,2)*mu(0)+mx(1,2)*mu(1)+mx(2,2)*mu(2));
        double denominator = mu(0)+mu(1)+mu(2);
        mu(0) = mu(0)/denominator;
        mu(1) = mu(1)/denominator;
        mu(2) = mu(2)/denominator;

        return mu;
    }

    std::pair<M,M> combine_state_and_covariance()//#TODO - проверить на безопасность!
    {
        M comb_state = mu(0)*estimator1.getState()+mu(1)*estimator2.getState()+mu(2)*estimator3.getState();
        M dX1 = estimator1.getState()-comb_state;
        M dX2 = estimator2.getState()-comb_state;
        M dX3 = estimator3.getState()-comb_state;
        M comb_covariance = mu(0)*(estimator1.getCovariance()+dX1*Utils::transpose(dX1))+
                            mu(1)*(estimator2.getCovariance()+dX2*Utils::transpose(dX2))+
                            mu(2)*(estimator3.getCovariance()+dX3*Utils::transpose(dX3));
        return std::make_pair(comb_state,comb_covariance);
    }

    std::pair<M,M> predict(double dt)
    {
        //calculate mixed states and covariances
        Mixes mxs = mixed_states_and_covariances();
        //set mixed states and covariaces
        estimator1.setState(mxs.X01);
        estimator1.setCovariance(mxs.P01);
        estimator2.setState(mxs.X02);
        estimator2.setCovariance(mxs.P02);
        estimator3.setState(mxs.X03);
        estimator3.setCovariance(mxs.P03);
        //predicts
        estimator1.predict(dt);
        estimator2.predict(dt);
        estimator3.predict(dt);
        //calculate combine state and covariance
        auto comb = combine_state_and_covariance();
        state = state_predict = comb.first;
        covariance = covariance_predict = comb.second;

        return std::make_pair(state,covariance);
    }

    std::pair<M,M> correct(const M& measurement)
    {
        //corrects
        estimator1.correct(measurement);
        estimator2.correct(measurement);
        estimator3.correct(measurement);
        //recalculate probabilities
        M z = Utils::transpose(measurement); //#TODO - не красиво транспонировать каждую матрицу
        M zp1 = Utils::transpose(estimator1.getMeasurementPredict());
        M zp2 = Utils::transpose(estimator2.getMeasurementPredict());
        M zp3 = Utils::transpose(estimator3.getMeasurementPredict());
        M Se1 = estimator1.getCovarianceOfMeasurementPredict();
        M Se2 = estimator2.getCovarianceOfMeasurementPredict();
        M Se3 = estimator3.getCovarianceOfMeasurementPredict();
        models_probability_recalculation(z,zp1,zp2,zp3,Se1,Se2,Se3);
        //calculate combine state and covariance
        auto comb = combine_state_and_covariance();
        state = comb.first;
        covariance = comb.second;

        return std::make_pair(state,covariance);
    }

public:
    M getState()const{return state;}
    M getCovariance()const{return covariance;}
    M getMU()const{return mu;}

};
}
