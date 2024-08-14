#pragma once

#include "ekf_eigen3.h"
#include "ekf.h"
#include <map>
#include <string>
#include <memory>
#include <map>
#include "models.h"

template<class M=Eigen::MatrixXd>
struct Measurement
{
    double timepoint;
    M point;
};

template<class M, class StateModel, class MeasureModel>
struct EstimatorInitEKFE_xyz_ct
{
    M R; M Q; M P0; M x0;
    StateModel SM;
    MeasureModel MM;

    std::unique_ptr<Estimator::EKFE<Eigen::MatrixXd,StateModel,MeasureModel>> make_estimator()
    {
        //std::cout << "R:" << std::endl << R << std::endl;
        return std::make_unique<Estimator::EKFE<Eigen::MatrixXd,StateModel,MeasureModel>>(x0,P0,Q,R);
    }

    EstimatorInitEKFE_xyz_ct(const Measurement<M>& measurement)
    {
        //Здесь пока задаются все общие параметры:
        //-------------------------------------------------------------------------
        double dt = 6; //откуда?
        double process_var = 0.01;//0.5; //откуда?
        //#CAUSE
        double process_var_w = 0.001;//0.00017;//1;//0.001; //откуда?
        double meas_std_decart = 100.;//300.;//300.;//1.; //откуда?
        double velo_std_decart = 200.;//30.;//1.;//30.;//1.; //откуда?
        //double process_var_w = 0.000001;
        //double meas_std_decart = 30.;
        //double velo_std_decart = 3.;
        //~

        //#CAUSE
        double w_var = Utils::rad2deg(0.392)*Utils::rad2deg(0.392);//0.098*0.098;//40;//0.4*180/M_PI;//2*0.4;//0.4;//1.;//0.1;//0.0007;
        //double w_var = 0.392*0.392;
        //~
        //-------------------------------------------------------------------------

        Models::GModel_XvXYvYZvZW<Eigen::MatrixXd> gm;

        M point = measurement.point;

        M Q0(4,4);
        Q0 << process_var,          0.,          0.,            0.,
                       0., process_var,          0.,            0.,
                       0.,          0., process_var,            0.,
                       0.,          0.,          0., process_var_w;

        M G = gm(dt);
        Q = G*Q0*Utils::transpose(G);
        //#CAUSE
        Q = Q0;
        //Q = G*Q0*Utils::transpose(G);
        //~

        R.resize(3,3);
        R << meas_std_decart*meas_std_decart,                              0.,                               0.,
                                          0., meas_std_decart*meas_std_decart,                               0.,
                                          0.,                              0.,  meas_std_decart*meas_std_decart;


        M Rvel(3,3);
        Rvel << velo_std_decart*velo_std_decart,                              0.,                               0.,
                                             0., velo_std_decart*velo_std_decart,                               0.,
                                             0.,                              0.,  velo_std_decart*velo_std_decart;
        M Hp(3,7);
        Hp << 1., 0., 0., 0., 0., 0., 0.,
              0., 0., 1., 0., 0., 0., 0.,
              0., 0., 0., 0., 1., 0., 0.;
        M Hv(3,7);
        Hv << 0., 1., 0., 0., 0., 0., 0.,
              0., 0., 0., 1., 0., 0., 0.,
              0., 0., 0., 0., 0., 1., 0.;

        x0 = Utils::transpose(Hp)*point;
        P0  = Utils::transpose(Hp)*R*Hp + Utils::transpose(Hv)*Rvel*Hv;
        P0(6,6) = w_var;
    }
};

template<class M, class StateModel, class MeasureModel, class GModel, class StateModelJ, class MeasureModelJ>
struct EstimatorInitEKF_xyz_ct
{
    M R; M Q0; M Q; M P0; M x0;
    StateModel SM;
    MeasureModel MM;

    std::unique_ptr<Estimator::EKF<Eigen::MatrixXd,StateModel,MeasureModel,GModel,StateModelJ,MeasureModelJ>> make_estimator()
    {
        //std::cout << "R:" << std::endl << R << std::endl;
        return std::make_unique<Estimator::EKF<Eigen::MatrixXd,StateModel,MeasureModel,GModel,StateModelJ,MeasureModelJ>>(x0,P0,Q,R);
    }

    EstimatorInitEKF_xyz_ct(const Measurement<M>& measurement)
    {
        //Здесь пока задаются все общие параметры:
        //-------------------------------------------------------------------------
        double dt = 6; //откуда?
        double process_var = 0.01;//0.5; //откуда?
        //#CAUSE
        double process_var_w = 0.001;//0.00017;//1;//0.001; //откуда?
        double meas_std_decart = 100.;//300.;//300.;//1.; //откуда?
        double velo_std_decart = 200.;//30.;//1.;//30.;//1.; //откуда?
        //double process_var_w = 0.000001;
        //double meas_std_decart = 30.;
        //double velo_std_decart = 3.;
        //~

        //#CAUSE
        double w_var = Utils::rad2deg(0.392)*Utils::rad2deg(0.392);//0.098*0.098;//40;//0.4*180/M_PI;//2*0.4;//0.4;//1.;//0.1;//0.0007;
        //double w_var = 0.392*0.392;
        //~
        //-------------------------------------------------------------------------

        GModel gm;

        M point = measurement.point;

        M Q0(4,4);
        Q0 << process_var,          0.,          0.,            0.,
                       0., process_var,          0.,            0.,
                       0.,          0., process_var,            0.,
                       0.,          0.,          0., process_var_w;

        M G = gm(dt);
        Q = G*Q0*Utils::transpose(G);
        //#CAUSE
        Q = Q0;
        //Q = G*Q0*Utils::transpose(G);
        //~

        R.resize(3,3);
        R << meas_std_decart*meas_std_decart,                              0.,                               0.,
                                          0., meas_std_decart*meas_std_decart,                               0.,
                                          0.,                              0.,  meas_std_decart*meas_std_decart;


        M Rvel(3,3);
        Rvel << velo_std_decart*velo_std_decart,                              0.,                               0.,
                                             0., velo_std_decart*velo_std_decart,                               0.,
                                             0.,                              0.,  velo_std_decart*velo_std_decart;
        M Hp(3,7);
        Hp << 1., 0., 0., 0., 0., 0., 0.,
              0., 0., 1., 0., 0., 0., 0.,
              0., 0., 0., 0., 1., 0., 0.;
        M Hv(3,7);
        Hv << 0., 1., 0., 0., 0., 0., 0.,
              0., 0., 0., 1., 0., 0., 0.,
              0., 0., 0., 0., 0., 1., 0.;

        x0 = Utils::transpose(Hp)*point;
        P0  = Utils::transpose(Hp)*R*Hp + Utils::transpose(Hv)*Rvel*Hv;
        P0(6,6) = w_var;
    }
};

template<class M, class StateModel, class MeasureModel, class GModel, class StateModelJ, class MeasureModelJ>
struct EstimatorInitEKF_xyz_ct_rad
{
    M R; M Q0; M Q; M P0; M x0;
    StateModel SM;
    MeasureModel MM;

    std::unique_ptr<Estimator::EKF<Eigen::MatrixXd,StateModel,MeasureModel,GModel,StateModelJ,MeasureModelJ>> make_estimator()
    {
        //std::cout << "R:" << std::endl << R << std::endl;
        return std::make_unique<Estimator::EKF<Eigen::MatrixXd,StateModel,MeasureModel,GModel,StateModelJ,MeasureModelJ>>(x0,P0,Q,R);
    }

    EstimatorInitEKF_xyz_ct_rad(const Measurement<M>& measurement)
    {
        //Здесь пока задаются все общие параметры:
        //-------------------------------------------------------------------------
        double dt = 6; //откуда?
        double process_var = 0.01;//0.5; //откуда?

        double process_var_w = 0.001*(M_PI/180.);//0.00017;//1;//0.001; //откуда?
        double meas_std_decart = 100.;//300.;//300.;//1.; //откуда?
        double velo_std_decart = 200.;//30.;//1.;//30.;//1.; //откуда?


        double w_var = 0.392*0.392;//0.098*0.098;//40;//0.4*180/M_PI;//2*0.4;//0.4;//1.;//0.1;//0.0007;
        //-------------------------------------------------------------------------

        GModel gm;

        M point = measurement.point;

        M Q0(4,4);
        Q0 << process_var,          0.,          0.,            0.,
                       0., process_var,          0.,            0.,
                       0.,          0., process_var,            0.,
                       0.,          0.,          0., process_var_w;

        M G = gm(dt);
        Q = G*Q0*Utils::transpose(G);
        //#CAUSE
        Q = Q0;
        //Q = G*Q0*Utils::transpose(G);
        //~

        R.resize(3,3);
        R << meas_std_decart*meas_std_decart,                              0.,                               0.,
                                          0., meas_std_decart*meas_std_decart,                               0.,
                                          0.,                              0.,  meas_std_decart*meas_std_decart;


        M Rvel(3,3);
        Rvel << velo_std_decart*velo_std_decart,                              0.,                               0.,
                                             0., velo_std_decart*velo_std_decart,                               0.,
                                             0.,                              0.,  velo_std_decart*velo_std_decart;
        M Hp(3,7);
        Hp << 1., 0., 0., 0., 0., 0., 0.,
              0., 0., 1., 0., 0., 0., 0.,
              0., 0., 0., 0., 1., 0., 0.;
        M Hv(3,7);
        Hv << 0., 1., 0., 0., 0., 0., 0.,
              0., 0., 0., 1., 0., 0., 0.,
              0., 0., 0., 0., 0., 1., 0.;

        x0 = Utils::transpose(Hp)*point;
        P0  = Utils::transpose(Hp)*R*Hp + Utils::transpose(Hv)*Rvel*Hv;
        P0(6,6) = w_var;
    }
};

template<class M, class EstimatorType, class EstimatorInit, class SM, class MM>
class Track
{
private:
    std::unique_ptr<EstimatorType> estimator;
    double timepoint;
public:
    Track(const Measurement<M>& m)
    {
        EstimatorInit ei(m);
        estimator = ei.make_estimator();
        timepoint = m.timepoint;
    }
    std::pair<M,M> step(const Measurement<M>& m)
    {
        double dt = m.timepoint - timepoint;
        timepoint = m.timepoint;

        M res_state;
        M res_covariance;

        auto p = estimator->predict(dt);
        res_state = p.first;
        res_covariance = p.second;

        auto c = estimator->correct(m.point);
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
    M GetState(){return estimator->GetState();}
    M GetCovariance(){return estimator->GetStateCovariance();}
};
