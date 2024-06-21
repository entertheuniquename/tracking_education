#pragma once
#include <memory>
#include "kf.h"
#include "ekf.h"
#include "models.h"
#include <math.h>

template<class M=Eigen::MatrixXd>
struct Measurement
{
    double timepoint;
    M point;
    //M measurement_noise; //TODO - перенести шумы в отметку
};

namespace Estimator
{

template<class M, class E1, class E2>
class IMM
{
private:
    M state;
    M covariance;
    M state_predict;
    M covariance_predict;
    E1 estimator1;
    E2 estimator2;
    M mu;
    M tp;
public:
    struct Mixes
    {
        M X01;
        M P01;
        M X02;
        M P02;
    };

    IMM(E1 e1, E2 e2, M m, M t):
        estimator1(e1),
        estimator2(e2),
        mu(m),
        tp(t){}

    IMM(M state,M covariance,M process_noise,M measureNoise, M m, M t):
        estimator1(state,covariance,process_noise,measureNoise),
        estimator2(state,covariance,process_noise,measureNoise),
        mu(m),
        tp(t){}

    M mixing_probability()//[1]
    {//#TODO - imm - размерности матриц автоматизировать. Автоматизировать количество фильтров(?)
        M mx(2,2);
        mx.setZero();
        M fi(1,2);
        fi.setZero();

        fi(0) = (mu(0)*tp(0,0)+mu(1)*tp(1,0));
        fi(1) = (mu(0)*tp(0,1)+mu(1)*tp(1,1));

        mx(0,0) = tp(0,0)*mu(0)/fi(0);
        mx(1,0) = tp(1,0)*mu(1)/fi(0);
        mx(0,1) = tp(0,1)*mu(0)/fi(1);
        mx(1,1) = tp(1,1)*mu(1)/fi(1);

        return mx;
    }

    Mixes mixed_states_and_covariances()//[2]
    {
        M mx = mixing_probability();
        Mixes mxs;

        M X1 = getStateE1();
        M X2 = getStateE2();
        M P1 = getCovarianceE1();
        M P2 = getCovarianceE2();

        mxs.X01 = mx(0,0)*X1+mx(1,0)*X2;
        mxs.X02 = mx(0,1)*X1+mx(1,1)*X2;

        M dX101 = X1-mxs.X01;
        M dX201 = X2-mxs.X01;
        M dX102 = X1-mxs.X02;
        M dX202 = X2-mxs.X02;

        mxs.P01 = (mx(0,0)*(P1+dX101*Utils::transpose(dX101))+
                   mx(1,0)*(P2+dX201*Utils::transpose(dX201)));
        mxs.P02 = (mx(0,1)*(P1+dX102*Utils::transpose(dX102))+
                   mx(1,1)*(P2+dX202*Utils::transpose(dX202)));

        return mxs;
    }

    double system_probability(M& z,M& ze,M& Se)
    {
        double n = 2;//#TODO - imm - размерность вектора измерений
        M residual = z-ze;
        M power = -0.5*residual*Utils::inverse(Se)*Utils::transpose(residual);
        return std::pow((1./2./M_PI),(n/2))/std::sqrt(Se.determinant())*std::exp(power.determinant());//#TODO - сверить с формулой!;
    }

    M models_probability_recalculation(M& z, M& ze1, M& ze2, M& Se1, M& Se2)
    {
        M mx = mixing_probability();
        mu(0) = system_probability(z,ze1,Se1)*(mx(0,0)*mu(0)+mx(1,0)*mu(1));
        mu(1) = system_probability(z,ze2,Se2)*(mx(0,1)*mu(0)+mx(1,1)*mu(1));
        double denominator = mu(0)+mu(1);
        mu(0) = mu(0)/denominator;
        mu(1) = mu(1)/denominator;

        return mu;
    }

    void combine_state_and_covariance()
    {
        state = mu(0)*getStateE1()+mu(1)*getStateE2();
        M dX1 = getStateE1()-state;
        M dX2 = getStateE2()-state;
        covariance = mu(0)*(getCovarianceE1()+dX1*Utils::transpose(dX1))+
            mu(1)*(getCovarianceE2()+dX2*Utils::transpose(dX2));
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
        //predicts
        estimator1.predict(dt);
        estimator2.predict(dt/*,nullptr*/,Models7::FCT_Jacobian<Eigen::MatrixXd>());//#TODO - почему нельзя вообще опустить аргумент?

        return std::make_pair(estimator1.getStatePredict(),estimator1.getCovariancePredict());//#BAD - это не правильные возвращаемые данные!
    }

    std::pair<M,M> correct(const M& measurement)
    {
        //corrects
        estimator1.correct(measurement);
        estimator2.correct(measurement);
        //recalculate probabilities
        M z = Utils::transpose(measurement); //#TODO - не красиво транспонировать каждую матрицу
        M zp1 = Utils::transpose(estimator1.getMeasurementPredict());
        M zp2 = Utils::transpose(estimator2.getMeasurementPredict());
        M Se1 = estimator1.getCovarianceOfMeasurementPredict();
        M Se2 = estimator2.getCovarianceOfMeasurementPredict();
        models_probability_recalculation(z,zp1,zp2,Se1,Se2);
        //calculate combine state and covariance
        combine_state_and_covariance();

        return std::make_pair(state,covariance);
    }

public:
    M getState()const{return state;}
    M getCovariance()const{return covariance;}
    M getStateE1()const{return estimator1.getState();}
    M getCovarianceE1()const{return estimator1.getCovariance();}
    M getStateE2()const{return estimator2.getState();}
    M getCovarianceE2()const{return estimator2.getCovariance();}
    M getMU()const{return mu;}

};
}
