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
        double TS = (std::sin(w)*T)/w;
        double TC = (1-std::cos(w)*T)/w;
        double S = std::sin(w)*T;
        double C = std::cos(w)*T;
        M F(7,7);
        F << 1., TS, 0., -TC, 0., 0., 0.,
             0.,  C, 0.,  -S, 0., 0., 0.,
             0., TC, 1.,  TS, 0., 0., 0.,
             0.,  S, 0.,   C, 0., 0., 0.,
             0., 0., 0.,  0., 1., T , 0.,
             0., 0., 0.,  0., 0., 1., 0.,
             0., 0., 0.,  0., 0., 0., 1.;
        return F*x;
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
    M operator()(const M& x)
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
}
