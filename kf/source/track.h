#pragma once

#include "kf_eigen3.h"
#include "ekf_eigen3.h"
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
    //M measurement_noise;
};

template<class M, class StateModel, class MeasureModel>
struct EstimatorInitKFE
{
    M R; M Q0; M Q; M P0; M x0;
    M SM;
    M MM;
    M GM;

    std::unique_ptr<Estimator::KFE<M,StateModel,MeasureModel>> make_estimator()
    {
        return std::make_unique<Estimator::KFE<M,StateModel,MeasureModel>>(x0,P0,SM,Q0,GM,MM,R);
    }

    EstimatorInitKFE(const Measurement<M>& measurement)
    {
        //Здесь пока задаются все общие параметры:
        //-------------------------------------------------------------------------
        double process_var = 1.;//откуда?
        double meas_std_decart = 300.;//откуда?
        double velo_std_decart = 30.;//откуда?
        //double meas_var_polar_ae = 0.0001;//откуда?
        //double meas_var_polar_r = 1.0;//откуда?
        double dt = 6.0;//откуда?
        //-------------------------------------------------------------------------

        StateModel sm;
        MeasureModel mm;

        M point = measurement.point;

        Q0.resize(3,3);
        Q0 << process_var,          0.,          0.,
                       0., process_var,          0.,
                       0.,          0., process_var;

        Models::GModel_XvXYvYZvZ<M> gm;

        GM = gm(dt);
        SM = sm(dt);
        MM = mm();

        M G = GM;
        Q = G*Q0*Utils::transpose(G);

        R.resize(3,3);
        R << meas_std_decart*meas_std_decart,                              0.,                               0.,
                                          0., meas_std_decart*meas_std_decart,                               0.,
                                          0.,                              0.,  meas_std_decart*meas_std_decart;

        M Rvel(3,3);
        Rvel << velo_std_decart*velo_std_decart,                              0.,                               0.,
                                             0., velo_std_decart*velo_std_decart,                               0.,
                                             0.,                              0.,  velo_std_decart*velo_std_decart;
        M Hp(3,6);
        Hp << 1., 0., 0., 0., 0., 0.,
              0., 0., 1., 0., 0., 0.,
              0., 0., 0., 0., 1., 0.;
        M Hv(3,6);
        Hv << 0., 1., 0., 0., 0., 0.,
              0., 0., 0., 1., 0., 0.,
              0., 0., 0., 0., 0., 1.;

         x0 = Utils::transpose(Hp)*point;

         P0  = Utils::transpose(Hp)*R*Hp + Utils::transpose(Hv)*Rvel*Hv;
    }
};

template<class M, class StateModel, class MeasureModel>
struct EstimatorInitKFEx
{
    M R; M Q0; M Q; M P0; M x0;
    StateModel SM;
    MeasureModel MM;
    M GM;

    std::unique_ptr<Estimator::KFEx<M,StateModel,MeasureModel>> make_estimator()
    {
        return std::make_unique<Estimator::KFEx<M,StateModel,MeasureModel>>(x0,P0,SM,Q0,GM,MM,R);
    }

    EstimatorInitKFEx(const Measurement<M>& measurement)
    {
        //Здесь пока задаются все общие параметры:
        //-------------------------------------------------------------------------
        double process_var = 1.;//откуда?
        double meas_std_decart = 300.;//откуда?
        double velo_std_decart = 30.;//откуда?
        double dt = 6.0;//откуда?
        //-------------------------------------------------------------------------

        M point = measurement.point;

        Q0.resize(3,3);
        Q0 << process_var,          0.,          0.,
                       0., process_var,          0.,
                       0.,          0., process_var;

        Models::GModel_XvXYvYZvZW<M> gm;

        GM = gm(dt);
        M G = GM;
        Q = G*Q0*Utils::transpose(G);

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
        x0(6,0) = 0.098;//#SET_DATA
        P0  = Utils::transpose(Hp)*R*Hp + Utils::transpose(Hv)*Rvel*Hv;
    }
};

template<class M, class StateModel, class MeasureModel>
struct EstimatorInitEKFE
{
    M R; M Q; M P0; M x0;
    StateModel SM;
    MeasureModel MM;

    std::unique_ptr<Estimator::EKFE<Eigen::MatrixXd,StateModel,MeasureModel>> make_estimator()
    {
        return std::make_unique<Estimator::EKFE<Eigen::MatrixXd,StateModel,MeasureModel>>(x0,P0,Q,R);
    }

    EstimatorInitEKFE(const Measurement<M>& measurement)
    {
        //Здесь пока задаются все общие параметры:
        //-------------------------------------------------------------------------
        double process_var = 1.;//откуда?
        double meas_std_decart = 300.;//откуда?
        double velo_std_decart = 30.;//откуда?
        double meas_var_polar_ae = 0.0001;//откуда?
        double meas_var_polar_r = 1.0;//откуда?
        double dt = 6.0;//откуда?
        //-------------------------------------------------------------------------

        Models::GModel_XvXYvYZvZ<Eigen::MatrixXd> gm;
        //StateModel sm;
        //MeasureModel mm;

        M point = measurement.point;

        M Q0(3,3);
        Q0 << process_var,          0.,          0.,
                       0., process_var,          0.,
                       0.,          0., process_var;

        M G = gm(dt);
        Q = G*Q0*Utils::transpose(G);

        R.resize(3,3);
        R << meas_var_polar_ae,                0.,                0.,
                            0., meas_var_polar_ae,                0.,
                            0.,                0.,  meas_var_polar_r;

        M Hp(3,6);
        Hp << 1., 0., 0., 0., 0., 0.,
              0., 0., 1., 0., 0., 0.,
              0., 0., 0., 0., 1., 0.;

        x0 = sph2cart(point);
        x0 = Utils::transpose(Hp)*x0;

        P0  = make_cartcov(point, R);
    }

    M rot_z(double a)
    {
        a = Utils::deg2rad(a);
        M R(3,3);
        R.setZero();
        double ca = std::cos(a);
        double sa = std::sin(a);
        R(0,0) = ca;
        R(0,1) = -sa;
        R(1,0) = sa;
        R(1,1) = ca;
        return R;
    }

    M rot_y(double a)
    {
        a = Utils::deg2rad(a);
        M R(3,3);
        R.setZero();
        double ca = std::cos(a);
        double sa = std::sin(a);
        R(0,0) = ca;
        R(0,2) = sa;
        R(2,0) = -sa;
        R(2,2) = ca;
        return R;
    }

    M sph2cart(M polar)
    {
        M decart(polar.rows(),polar.cols());
        decart.setZero();
        for(int i=0;i<polar.cols();i++)
        {
            decart(0, i) = polar(2, i) * std::cos(polar(1, i)) * std::cos(polar(0, i));
            decart(1, i) = polar(2, i) * std::sin(polar(1, i)) * std::cos(polar(0, i));
            decart(2, i) = polar(2, i) * std::sin(polar(0, i));
        }
        return decart;
    }

    std::pair<M, M> sph2cartcov(M sphCov, double az, double el, double r)
    {
        int pr = 2;
        int pa = 0;
        int pe = 1;
        int pvr= 3;

        double rngSig = std::sqrt(sphCov(pr, pr));
        double azSig  = std::sqrt(sphCov(pa, pa));
        double elSig  = std::sqrt(sphCov(pe, pe));


        M Rpos(3,3);
        Rpos << rngSig*rngSig,                                                                                                            0.,                                                  0.,
                           0., (r*std::cos(Utils::deg2rad(el))*Utils::deg2rad(azSig))*(r*std::cos(Utils::deg2rad(el))*Utils::deg2rad(azSig)),                                                  0.,
                           0.,                                                                                                            0., (r*Utils::deg2rad(elSig))*(r*Utils::deg2rad(elSig));
        M rot = rot_z(az)*Utils::transpose(rot_y(el));
        M posCov = rot*Rpos*Utils::transpose(rot);
        M velCov;
        if(sphCov.rows()==4 && sphCov.cols()==4)
        {
            double rrSig = std::sqrt(sphCov(pvr, pvr));
            double crossVelSig = 10.;
            M Rvel(3,3);
            Rvel << rrSig*rrSig,                      0.,                      0.,
                             0., crossVelSig*crossVelSig,                      0.,
                             0.,                      0., crossVelSig*crossVelSig;
            velCov = rot*Rvel*Utils::transpose(rot);
        }
        else
        {
            velCov.resize(3,3);
            velCov << 100.,   0.,   0.,
                        0., 100.,   0.,
                        0.,   0., 100.;
        }

        return std::make_pair(posCov, velCov);
    }

    M make_cartcov(M meas, M covMeas)
    {
        double r  = meas(2,0);
        double az = Utils::rad2deg(meas(1,0));
        double el = Utils::rad2deg(meas(0,0));
        M sphCov(3,3);
        sphCov << covMeas(0,0),           0.,           0.,
                            0., covMeas(2,2),           0.,
                            0.,           0., covMeas(1,1);
        std::pair<M, M> x = sph2cartcov(sphCov, az, el, r);
        M Hp(3,6);
        Hp << 1., 0., 0., 0., 0., 0.,
              0., 0., 1., 0., 0., 0.,
              0., 0., 0., 0., 1., 0.;
        M Hv(3,6);
        Hv << 0., 1., 0., 0., 0., 0.,
              0., 0., 0., 1., 0., 0.,
              0., 0., 0., 0., 0., 1.;
        return Utils::transpose(Hp)*x.first*Hp + Utils::transpose(Hv)*x.second*Hv;
    }
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
        double process_var_w = 0.000001;//0.00017;//1;//0.001; //откуда?
        double meas_std_decart = 30.;//300.;//300.;//1.; //откуда?
        double velo_std_decart = 3.;//30.;//1.;//30.;//1.; //откуда?

        double w_var = 0.392*0.392;//0.098*0.098;//40;//0.4*180/M_PI;//2*0.4;//0.4;//1.;//0.1;//0.0007;
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
