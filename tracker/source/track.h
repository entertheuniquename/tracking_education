#pragma once

#include "filter.h"
#include "measurement.h"
#include "utils.h"
#include <cmath>

namespace Tracker
{

template<class M, class EstimatorType, class MeasurementType, class ...Types>
class EstimatorInitializator10
{
public:
    Estimator::Filter<M>* operator()(MeasurementType m, Types ...args)
    {
        double std_proc_x = 1.;//#TODO - в инит-файл?
        double std_proc_y = 1.;//#TODO - в инит-файл?
        double std_proc_z = 1.;//#TODO - в инит-файл?
        double std_proc_w = 0.01;//#TODO - в инит-файл?

        double meas_std = 300;//#TEMP //#TODO - вынести куда-то!!!
        double velo_std = 30;//#TEMP //#TODO - вынести куда-то!!!
        double acc_std = 3;//#TEMP //#TODO - вынести куда-то!!!
        double w_std = 0.392;//#TEMP //#TODO - вынести куда-то!!!

        M startState(10,1);
        startState << m.x(),m.vx(),m.ax(),m.y(),m.vy(),m.ay(),m.z(),m.vz(),m.az(),m.w();

        //std::cout << "imm.initializator.startState " << Utils::transpose(startState) << std::endl;

        M measNoise(3,3);
        measNoise << std::pow(/*m.std_meas_x()*/meas_std,2), 0., 0.,//#TEMP
                     0., std::pow(/*m.std_meas_y()*/meas_std,2), 0.,//#TEMP
                     0., 0., std::pow(/*m.std_meas_z()*/meas_std,2);//#TEMP
        M measVeloNoise(3,3);
        measVeloNoise << std::pow(/*m.std_meas_velo_x(),*/velo_std,2), 0., 0.,//#TEMP
                         0., std::pow(/*m.std_meas_velo_y()*/velo_std,2), 0.,//#TEMP
                         0., 0., std::pow(/*m.std_meas_velo_z()*/velo_std,2);//#TEMP
        M measAccNoise(3,3);
        measAccNoise << std::pow(/*m.std_meas_velo_x(),*/acc_std,2), 0., 0.,//#TEMP
                         0., std::pow(/*m.std_meas_velo_y()*/acc_std,2), 0.,//#TEMP
                         0., 0., std::pow(/*m.std_meas_velo_z()*/acc_std,2);//#TEMP
        M processNoise(4,4);
        processNoise << std::pow(std_proc_x,2), 0., 0., 0.,
                        0., std::pow(std_proc_y,2), 0., 0.,
                        0., 0., std::pow(std_proc_z,2), 0.,
                        0., 0., 0., std::pow(std_proc_w,2);
        M Hp(3,10);
        Hp << 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 0., 1., 0., 0., 0.;
        M Hv(3,10);
        Hv << 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 0., 0., 1., 0., 0.;
        M Ha(3,10);
        Ha << 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 0., 0., 0., 1., 0.;
        M Hw(1,10);
        Hw << 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.;
        M startCovariance  = Utils::transpose(Hp)*measNoise*Hp + Utils::transpose(Hv)*measVeloNoise*Hv + Utils::transpose(Ha)*measAccNoise*Ha + Utils::transpose(Hw)*std::pow(w_std,2)*Hw;

        //std::cout << "startState" << std::endl << startState << std::endl;
        //std::cout << "startCovariance" << std::endl << startCovariance << std::endl;
        //std::cout << "processNoise" << std::endl << processNoise << std::endl;
        //std::cout << "measNoise" << std::endl << measNoise << std::endl;

        return new EstimatorType(startState,startCovariance,processNoise,measNoise,args...);
    }
};

template<class M, class EstimatorType, class MeasurementType, class ...Types>
class EstimatorInitializator10_IMM3
{
public:
    Estimator::Filter<M>* operator()(MeasurementType m, Types ...args)
    {
        Eigen::MatrixXd mu(1,3);
        mu << 0.3333, 0.3333, 0.3333;
        Eigen::MatrixXd trans(3,3);
        trans << 0.95, 0.025, 0.025,
                 0.025, 0.95, 0.025,
                 0.025, 0.025, 0.95;

        double meas_std = 300;//#TEMP //#TODO - вынести куда-то!!!
        double velo_std = 30;//#TEMP //#TODO - вынести куда-то!!!
        double acc_std = 3;//#TEMP //#TODO - вынести куда-то!!!
        double w_std = 0.392;//#TEMP //#TODO - вынести куда-то!!!


        double std_proc_x = 1.;//#TODO - в инит-файл?
        double std_proc_y = 1.;//#TODO - в инит-файл?
        double std_proc_z = 1.;//#TODO - в инит-файл?
        double std_proc_w = 0.01;//#TODO - в инит-файл?

        M startState(10,1);
        startState << m.x(),m.vx(),m.ax(),m.y(),m.vy(),m.ay(),m.z(),m.vz(),m.az(),m.w();

        //std::cout << "imm.initializator.startState " << Utils::transpose(startState) << std::endl;

        M measNoise(3,3);
        measNoise << std::pow(/*m.std_meas_x()*/meas_std,2), 0., 0.,//#TEMP
                     0., std::pow(/*m.std_meas_y()*/meas_std,2), 0.,//#TEMP
                     0., 0., std::pow(/*m.std_meas_z()*/meas_std,2);//#TEMP
        M measVeloNoise(3,3);
        measVeloNoise << std::pow(/*m.std_meas_velo_x(),*/velo_std,2), 0., 0.,//#TEMP
                         0., std::pow(/*m.std_meas_velo_y()*/velo_std,2), 0.,//#TEMP
                         0., 0., std::pow(/*m.std_meas_velo_z()*/velo_std,2);//#TEMP
        M measAccNoise(3,3);
        measAccNoise << std::pow(/*m.std_meas_velo_x(),*/acc_std,2), 0., 0.,//#TEMP
                         0., std::pow(/*m.std_meas_velo_y()*/acc_std,2), 0.,//#TEMP
                         0., 0., std::pow(/*m.std_meas_velo_z()*/acc_std,2);//#TEMP
        M processNoise(4,4);
        processNoise << std::pow(std_proc_x,2), 0., 0., 0.,
                        0., std::pow(std_proc_y,2), 0., 0.,
                        0., 0., std::pow(std_proc_z,2), 0.,
                        0., 0., 0., std::pow(std_proc_w,2);
        M Hp(3,10);
        Hp << 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 0., 1., 0., 0., 0.;
        M Hv(3,10);
        Hv << 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 0., 0., 1., 0., 0.;
        M Ha(3,10);
        Ha << 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 0., 0., 0., 1., 0.;
        M Hw(1,10);
        Hw << 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.;
        M startCovariance  = Utils::transpose(Hp)*measNoise*Hp + Utils::transpose(Hv)*measVeloNoise*Hv + Utils::transpose(Ha)*measAccNoise*Ha + Utils::transpose(Hw)*std::pow(w_std,2)*Hw;

        //std::cout << "startState" << std::endl << startState << std::endl;
        //std::cout << "startCovariance" << std::endl << startCovariance << std::endl;
        //std::cout << "processNoise" << std::endl << processNoise << std::endl;
        //std::cout << "measNoise" << std::endl << measNoise << std::endl;

        return new EstimatorType(mu,trans,startState,startCovariance,processNoise,measNoise);
    }
};

template<class M, class EstimatorType, class MeasurementType, class ...Types>
class EstimatorInitializator4
{
public:
    Estimator::Filter<M>* operator()(const MeasurementType& m, Types ...args)
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

        return new EstimatorType(startState,startCovariance,processNoise,measNoise, args...);
    }
};

template<class M, class MeasurementType, class EstimatorInitType>
class Track
{
    enum struct STATE{UNKNOWN=0,NEW=1,ACTUAL=2,OLD=3};
private:
    Estimator::Filter<M>* estimator;
    //double price;//#TODO
    double timepoint;
    bool is_init;
    std::vector<bool> pass_meas;
    STATE state;
public:
    long long unsigned int id;

    Track():
        timepoint(0.),
        estimator(nullptr),
        //price(0.),//#TODO
        is_init(false),
        state(STATE::UNKNOWN)
    {}

    void counter(bool b_measurement)
    {
        switch (b_measurement) {
        case true:
            pass_meas.push_back(true);
            break;
        case false:
            pass_meas.push_back(false);
            break;
        }

        int start_unit = pass_meas.size()-4;
        if(start_unit<0)start_unit=0;
        int end_unit=pass_meas.size()-1;

        int work_units = (end_unit-start_unit+1);

        int sum=0;

        for(int i=start_unit;i<=end_unit;i++)
            sum+=pass_meas.at(i);


        if(state==STATE::NEW && work_units>=3 && sum>=3./4.)
            state=STATE::ACTUAL;
        if(state==STATE::ACTUAL && work_units>=3 && sum<=1./4.)
            state=STATE::OLD;
    }

    template<class EstimatorInitializator=EstimatorInitType, class ...Types>
    void initialization(MeasurementType m, Types ...args)
    {
        estimator=EstimatorInitializator()(m,args...);
        timepoint=m.timepoint();
        is_init=true;
        state = STATE::NEW;

        counter(true);
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

        counter(true);

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

        counter(false);

        return std::make_pair(res_state,res_covariance);
    }
    M getState(){return estimator->getState();}
    M getCovariance(){return estimator->getCovariance();}
    M getMeasurementPredict(){return estimator->getMeasurementPredict();}
    std::pair<M,M> getMeasurementPredictData(double dt){return estimator->getMeasurementPredictData(dt);}//#TODO
    M getCovarianceOfMeasurementPredict(){return estimator->getCovarianceOfMeasurementPredict();}
    bool isInit(){return is_init;}
    double getTimePoint(){return timepoint;}
    double getBigGate(double maxVel, double dt){return maxVel*dt;}
    double getGate(/*double Pd, double beta, */M S)
    {
        double Pd = 0.95;
        int m = S.rows();
        double beta = 0.1;
        return (2*std::log(Pd/((1-Pd)*std::pow(2*M_PI,m/2)*beta*std::sqrt(S.determinant()))));
    }
    //double getPrice()const{return price;}//#TODO
    bool isLastMeas(){return pass_meas.back();}
};
}
