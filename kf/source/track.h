#pragma once

#include "filter.h"
#include "measurement.h"
#include "utils.h"
#include <cmath>

namespace Tracker
{

template<class M, class EstimatorType, class MeasurementType>
class EstimatorInitializator10
{
public:
    Estimator::Filter<M>* operator()(MeasurementType m)
    {
        double std_proc_x = 1.;//#TODO - в инит-файл?
        double std_proc_y = 1.;//#TODO - в инит-файл?
        double std_proc_z = 1.;//#TODO - в инит-файл?
        double std_proc_w = 0.01;//#TODO - в инит-файл?

        M startState(10,1);
        startState << m.x(),m.vx(),m.ax(),m.y(),m.vy(),m.ay(),m.z(),m.vz(),m.az(),m.w();
        M measNoise(3,3);
        measNoise << std::pow(m.std_meas_x(),2), 0., 0.,
                     0., std::pow(m.std_meas_y(),2), 0.,
                     0., 0., std::pow(m.std_meas_z(),2);
        M measVeloNoise(3,3);
        measVeloNoise << std::pow(m.std_meas_velo_x(),2), 0., 0.,
                         0., std::pow(m.std_meas_velo_y(),2), 0.,
                         0., 0., std::pow(m.std_meas_velo_z(),2);
        M processNoise(3,3);
        processNoise << std::pow(std_proc_x,2), 0., 0.,
                        0., std::pow(std_proc_y,2), 0.,
                        0., 0., std::pow(std_proc_z,2);
        M Hp(3,6);
        Hp << 1., 0., 0., 0., 0., 0.,
              0., 0., 1., 0., 0., 0.,
              0., 0., 0., 0., 1., 0.;
        M Hv(3,6);
        Hv << 0., 1., 0., 0., 0., 0.,
              0., 0., 0., 1., 0., 0.,
              0., 0., 0., 0., 0., 1.;
        M startCovariance  = Utils::transpose(Hp)*measNoise*Hp + Utils::transpose(Hv)*measVeloNoise*Hv;

        return new EstimatorType(startState,startCovariance,processNoise,measNoise);
    }
};

template<class M, class EstimatorType, class MeasurementType>
class EstimatorInitializator4
{
public:
    Estimator::Filter<M>* operator()(const MeasurementType& m)
    {
        double std_proc_x = 1.;
        double std_proc_y = 1.;

        M startState(4,1);//#TODO - сделать не в лоб
        startState << m.x(),m.vx(),m.y(),m.vy();//#TODO - сделать не в лоб
        M measNoise(2,2);
        measNoise << std::pow(m.std_meas_x(),2), 0.,
                     0., std::pow(m.std_meas_y(),2);
        M measVeloNoise(2,2);
        measVeloNoise << std::pow(m.std_meas_velo_x(),2), 0.,
                         0., std::pow(m.std_meas_velo_y(),2);
        M processNoise(2,2);
        processNoise << std::pow(std_proc_x,2), 0.,
                        0., std::pow(std_proc_y,2);
        M Hp(2,4);
        Hp << 1., 0., 0., 0.,
              0., 0., 1., 0.;
        M Hv(2,4);
        Hv << 0., 1., 0., 0.,
              0., 0., 0., 1.;
        M startCovariance  = Utils::transpose(Hp)*measNoise*Hp + Utils::transpose(Hv)*measVeloNoise*Hv;

        return new EstimatorType(startState,startCovariance,processNoise,measNoise);
    }
};

template<class M, class MeasurementType, class EstimatorInitType>
class Track
{
private:
    Estimator::Filter<M>* estimator;
    //double price;//#TODO
    double timepoint;
    bool is_init;
public:
    long long unsigned int id;
    Track():
        timepoint(0.),
        estimator(nullptr),
        //price(0.),//#TODO
        is_init(false)
    {}

    template<class EstimatorInitializator=EstimatorInitType>
    void initialization(MeasurementType m)
    {
        estimator=EstimatorInitializator()(m);
        timepoint=m.timepoint();
        is_init=true;
    }
    std::pair<M,M> step(const MeasurementType& m)
    {
        double dt = m.timepoint() - timepoint;
        timepoint = m.timepoint();

        M res_state;
        M res_covariance;

        auto p = estimator->predict(dt);
        res_state = p.first;
        res_covariance = p.second;

        auto c = estimator->correct(m.get());
        res_state = c.first;
        res_covariance = c.second;

        return std::make_pair(res_state,res_covariance);
    }
    std::pair<M,M> step(double t)
    {
        double dt = t - timepoint;
        timepoint = t;
        M res_state;
        M res_covariance;

        auto p = estimator->predict(dt);
        res_state = p.first;
        res_covariance = p.second;

        return std::make_pair(res_state,res_covariance);
    }
    M getState(){return estimator->getState();}
    M getCovariance(){return estimator->getCovariance();}
    M getMeasurementPredict(){return estimator->getMeasurementPredict();}
    std::pair<M,M> getMeasurementPredictData(const double& dt){return estimator->getMeasurementPredictData(dt);}//#TODO
    M getCovarianceOfMeasurementPredict(){return estimator->getCovarianceOfMeasurementPredict();}
    bool isInit(){return is_init;}
    double getTimePoint(){return timepoint;}
    //double getPrice()const{return price;}//#TODO
};
}
