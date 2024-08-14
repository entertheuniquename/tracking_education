#pragma once

#include <math.h>
#include "utils.h"

namespace Models
{
//-----------------------------------------------------------------
template <class M>
struct StateModel_CV
{
    M operator()(const M& x,double T)
    {
        M F(6,6);
        F << 1., T , 0., 0., 0., 0.,
             0., 1., 0., 0., 0., 0.,
             0., 0., 1., T , 0., 0.,
             0., 0., 0., 1., 0., 0.,
             0., 0., 0., 0., 1., T ,
             0., 0., 0., 0., 0., 1.;
        return F*x;
    }
    M operator()(double T)
    {
        M F(6,6);
        F << 1., T , 0., 0., 0., 0.,
             0., 1., 0., 0., 0., 0.,
             0., 0., 1., T , 0., 0.,
             0., 0., 0., 1., 0., 0.,
             0., 0., 0., 0., 1., T ,
             0., 0., 0., 0., 0., 1.;
        return F;
    }
};
//-----------------------------------------------------------------
template <class M>
struct StateModel_CT
{
    enum class POSITION{X=0,VX=1,Y=2,VY=3,Z=4,VZ=5,W=6};
    M operator()(const M& x,double T)
    {
        double w = x(static_cast<int>(POSITION::W));
        if(w==0)//std::numeric_limits<double>::epsilon(); - min double //#TODO
            w=Utils::eps();
        //std::cout << "w:" << w << std::endl;
        double S = std::sin(w*T);
        double C = std::cos(w*T);
        double TS = S/w;
        double TC = (1-C)/w;

        M F(7,7);
        F << 1., TS, 0.,-TC, 0., 0., 0.,
             0.,  C, 0., -S, 0., 0., 0.,
             0., TC, 1., TS, 0., 0., 0.,
             0.,  S, 0.,  C, 0., 0., 0.,
             0., 0., 0., 0., 1., T , 0.,
             0., 0., 0., 0., 0., 1., 0.,
             0., 0., 0., 0., 0., 0., 1.;
        return F*x;
    }
};
template <class M>
struct StateModel_CT_Jacobian
{
    enum class POSITION{X=0,VX=1,Y=2,VY=3,Z=4,VZ=5,W=6};
    M operator()(const M& x,double T)
    {
        double vx = x(static_cast<int>(POSITION::VX));
        double vy = x(static_cast<int>(POSITION::VY));
        double w = x(static_cast<int>(POSITION::W));
        double t = T;

        if(w==0)
            w=Utils::eps();

        M J(7,7);
        J.setZero();

        J(0,0) = 1.;
        J(0,1) = std::sin(w*t)/w;
        J(0,2) = 0.;
        J(0,3) = (std::cos(w*t)-1)/w;
        J(0,4) = 0.;
        J(0,5) = 0.;
        J(0,6) = (t*vx*std::cos(w*t)/w) - (t*vy*std::sin(w*t)/w) - (vx*std::sin(w*t)/std::pow(w,2)) - (vy*(std::cos(w*t)-1)/std::pow(w,2));

        J(1,0) = 0.;
        J(1,1) = std::cos(w*t);
        J(1,2) = 0.;
        J(1,3) = -std::sin(w*t);
        J(1,4) = 0.;
        J(1,5) = 0.;
        J(1,6) = -t*vx*std::sin(w*t) - t*vy*std::cos(w*t);

        J(2,0) = 0.;
        J(2,1) = (1-std::cos(w*t))/w;
        J(2,2) = 1.;
        J(2,3) = std::sin(w*t)/w;
        J(2,4) = 0.;
        J(2,5) = 0.;
        J(2,6) = (t*vx*std::sin(w*t)/w) + (t*vy*std::cos(w*t)/w) - (vx*(1-std::cos(w*t))/std::pow(w,2)) - (vy*std::sin(w*t)/std::pow(w,2));

        J(3,0) = 0.;
        J(3,1) = std::sin(w*t);
        J(3,2) = 0.;
        J(3,3) = std::cos(w*t);
        J(3,4) = 0.;
        J(3,5) = 0.;
        J(3,6) = t*vx*std::cos(w*t) - t*vy*std::sin(w*t);

        J(4,0) = 0.;
        J(4,1) = 0.;
        J(4,2) = 0.;
        J(4,3) = 0.;
        J(4,4) = 1.;
        J(4,5) = t;
        J(4,6) = 0.;

        J(5,0) = 0.;
        J(5,1) = 0.;
        J(5,2) = 0.;
        J(5,3) = 0.;
        J(5,4) = 0.;
        J(5,5) = 1.;
        J(5,6) = 0.;

        J(6,0) = 0.;
        J(6,1) = 0.;
        J(6,2) = 0.;
        J(6,3) = 0.;
        J(6,4) = 0.;
        J(6,5) = 0.;
        J(6,6) = 1.;

        return J;
    }
};
//-----------------------------------------------------------------
//#CAUSE
template <class M>
struct StateModel_CT_Deg
{
    enum class POSITION{X=0,VX=1,Y=2,VY=3,Z=4,VZ=5,W=6};
    M operator()(const M& x,double T)
    {
        double w = x(static_cast<int>(POSITION::W));
        w = Utils::deg2rad(w);

        if(w==0)//std::numeric_limits<double>::epsilon(); - min double //#TODO
            w=Utils::eps();
        //std::cout << "w:" << w << std::endl;
        double S = std::sin(w*T);
        double C = std::cos(w*T);
        double TS = S/w;
        double TC = (1-C)/w;


        M F(7,7);
        F << 1., TS, 0.,-TC, 0., 0., 0.,
             0.,  C, 0., -S, 0., 0., 0.,
             0., TC, 1., TS, 0., 0., 0.,
             0.,  S, 0.,  C, 0., 0., 0.,
             0., 0., 0., 0., 1., T , 0.,
             0., 0., 0., 0., 0., 1., 0.,
             0., 0., 0., 0., 0., 0., 1.;
        return F*x;
    }
};
//
//~
template <class M>
struct StateModel_CT_Deg_Jacobian
{
    enum class POSITION{X=0,VX=1,Y=2,VY=3,Z=4,VZ=5,W=6};
    M operator()(const M& state,double dt)
    {
        size_t px  =size_t(POSITION::X ),
               pvx =size_t(POSITION::VX),
               py  =size_t(POSITION::Y ),
               pvy =size_t(POSITION::VY),
               po  =size_t(POSITION::W ),
               pz  =size_t(POSITION::Z ),
               pvz =size_t(POSITION::VZ);

        double dt2 = dt * dt / 2.;

        double omega = Utils::deg2rad(state(po));
        double eps = 4*std::numeric_limits<double>::epsilon();

        int K = 7;

        M jac = Utils::zeros(K, K);

        if (std::abs(omega) > std::sqrt(eps)) {

            double WT  = omega * dt;
            double CWT = cos(WT);
            double SWT = sin(WT);

            jac(px, px) = 1.;

            jac(px,  pvx) = SWT / omega;
            jac(pvx, pvx) = CWT;
            jac(py,  pvx) = (1. - CWT) / omega;
            jac(pvy, pvx) = SWT;

            jac(py, py) = 1.;

            jac(px,  pvy) = -(1. - CWT) / omega;
            jac(pvx, pvy) = -SWT;
            jac(py,  pvy) = SWT / omega;
            jac(pvy, pvy) = CWT;

            jac(px,  po) = Utils::deg2rad(((WT*CWT - SWT) * state(pvx) + (1 - CWT - WT*SWT) * state(pvy)) / std::pow(omega, 2.));
            jac(pvx, po) = Utils::deg2rad((-state(pvx) * SWT - state(pvy) * CWT) * dt);
            jac(py,  po) = Utils::deg2rad((WT*(state(pvx) * SWT + state(pvy) * CWT) - (state(pvx) * (1-CWT) + state(pvy) * SWT)) / std::pow(omega, 2.));
            jac(pvy, po) = Utils::deg2rad((state(pvx) * CWT - state(pvy) * SWT) * dt);

            jac(po, po) = 1.;

        } else {

            jac(px, px ) = 1.;
            jac(px, pvx) = dt;
            jac(pvx,pvx) = 1.;

            jac(py, py ) = 1.;
            jac(py, pvy) = dt;
            jac(pvy,pvy) = 1.;

            jac(po, po) = 1.;

            jac(px, po) = Utils::deg2rad(-state(pvy) * dt2);
            jac(pvx,po) = Utils::deg2rad(-state(pvy) * dt );
            jac(py, po) = Utils::deg2rad( state(pvx) * dt2);
            jac(pvy,po) = Utils::deg2rad( state(pvx) * dt );
        }

        jac(pz,  pz ) = 1.;
        jac(pz,  pvz) = dt;
        jac(pvz, pvz) = 1.;

        return jac;
    }

};
//-----------------------------------------------------------------
template <class M>
struct MeasureModel_XvXYvYZvZ_EAR
{
    M operator()(const M& x, const M& z = M{})
    {
        enum class POSITION{X=0,VX=1,Y=2,VY=3,Z=4,VZ=5};
        double X = x(static_cast<int>(POSITION::X));
        double Y = x(static_cast<int>(POSITION::Y));
        double Z = x(static_cast<int>(POSITION::Z));
        double elev = atan2(Z, sqrt(Y*Y+X*X));
        double angle = atan2(Y,X);
        double range = sqrt(X*X+Y*Y+Z*Z);
        if (!(z.cols()==0) && !(z.rows()==0))
        {
            angle = z(1) + Utils::ComputeAngleDifference(angle, z(1));
        }
        M r(1,3);
        r << elev, angle, range;
        return r.transpose();
}
};
//-----------------------------------------------------------------
template <class M>
struct MeasureModel_XvXYvYZvZ_XYZ
{
    M operator()(const M& x, const M& z = M{})//#TEMP z - для универсализации. Нужно по-другому обыграть.
    {
        M H(3,6);
        H << 1., 0., 0., 0., 0., 0.,
             0., 0., 1., 0., 0., 0.,
             0., 0., 0., 0., 1., 0.;
        return H*x;
    }
    M operator()()
    {
        M H(3,6);
        H << 1., 0., 0., 0., 0., 0.,
             0., 0., 1., 0., 0., 0.,
             0., 0., 0., 0., 1., 0.;
        return H;
    }
};
//-----------------------------------------------------------------
template <class M>
struct MeasureModel_XvXYvYZvZW_XYZ
{
    M operator()(const M& x, const M& z = M{})//#TEMP z - для универсализации. Нужно по-другому обыграть.
    {
        M H(3,7);
        H << 1., 0., 0., 0., 0., 0., 0.,
             0., 0., 1., 0., 0., 0., 0.,
             0., 0., 0., 0., 1., 0., 0.;
        return H*x;
    }
    M operator()()
    {
        M H(3,7);
        H << 1., 0., 0., 0., 0., 0., 0.,
             0., 0., 1., 0., 0., 0., 0.,
             0., 0., 0., 0., 1., 0., 0.;
        return H;
    }
};
//-----------------------------------------------------------------
template <class M>
struct MeasureModel_XvXYvYZvZW_XYZ_Jacobian
{
    M operator()(const M& x, const M& z = M{})//#TEMP z - для универсализации. Нужно по-другому обыграть.
    {
        M H(3,7);
        H << 1., 0., 0., 0., 0., 0., 0.,
             0., 0., 1., 0., 0., 0., 0.,
             0., 0., 0., 0., 1., 0., 0.;
        return H*x;
    }
    M operator()()
    {
        M H(3,7);
        H << 1., 0., 0., 0., 0., 0., 0.,
             0., 0., 1., 0., 0., 0., 0.,
             0., 0., 0., 0., 1., 0., 0.;
        return H;
    }
};
//-----------------------------------------------------------------
template <class M>
struct GModel_XvXYvYZvZ
{
    M operator()(double T)
    {
        M G(6,3);
        G <<   T*T/2.,       0.,       0.,
                   T ,       0.,       0.,
                   0.,   T*T/2.,       0.,
                   0.,       T ,       0.,
                   0.,       0.,   T*T/2.,
                   0.,       0.,       T ;
        return G;
    }
};
//-----------------------------------------------------------------
template <class M>
struct GModel_XvXYvYZvZW
{
    M operator()(double T)
    {
        M G(7,4);
        G <<   T*T/2.,       0.,       0.,       0.,
                   T ,       0.,       0.,       0.,
                   0.,   T*T/2.,       0.,       0.,
                   0.,       T ,       0.,       0.,
                   0.,       0.,   T*T/2.,       0.,
                   0.,       0.,       T ,       0.,
                   0.,       0.,       0.,       T ;
        return G;
    }
};
//-----------------------------------------------------------------
}
