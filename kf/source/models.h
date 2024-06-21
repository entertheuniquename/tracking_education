#pragma once

#include <math.h>
#include "utils.h"

namespace Models7
{
enum class POSITION_X{X=0,VX=1,Y=2,VY=3,Z=4,VZ=5,W=6};
enum class POSITION_Z{X=0,Y=1,Z=2};
//-----------------------------------------------------------------
template <class M>
struct FCV
{
    M operator()(const M& x,double T)
    {
        M F(7,7);
        F << 1., T , 0., 0., 0., 0., 0.,
             0., 1., 0., 0., 0., 0., 0.,
             0., 0., 1., T , 0., 0., 0.,
             0., 0., 0., 1., 0., 0., 0.,
             0., 0., 0., 0., 1., T , 0.,
             0., 0., 0., 0., 0., 1., 0.,
             0., 0., 0., 0., 0., 0., 1.;
        return F*x;
    }
};
//-----------------------------------------------------------------
template <class M>
struct FCT
{
    M operator()(const M& x,double T)
    {
        double w = x(static_cast<int>(POSITION_X::W));
        if(w==0)
            w=Utils::eps();

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
//-----------------------------------------------------------------
template <class M>
struct FCV_Jacobian
{
    M operator()(double T)
    {
        M J(7,7);
        J.setZero();
        J << 1.,T ,0.,0.,0.,0.,0.,
             0.,1.,0.,0.,0.,0.,0.,
             0.,0.,1.,T ,0.,0.,0.,
             0.,0.,0.,1.,0.,.0,0.,
             0.,0.,0.,0.,1.,T ,0.,
             0.,0.,0.,0.,0.,1.,0.,
             0.,0.,0.,0.,0.,0.,1.;
        return J;
    }
};
//-----------------------------------------------------------------
template <class M>
struct FCT_Jacobian
{
    M operator()(const M& x,double T)
    {
        double vx = x(static_cast<int>(POSITION_X::VX));
        double vy = x(static_cast<int>(POSITION_X::VY));
        double w = x(static_cast<int>(POSITION_X::W));
        double t = T;

        if(w==0)
            w=Utils::eps();

        M J(7,7);
        J.setZero();

        double J00 = 1.;
        double J01 = std::sin(w*t)/w;
        double J02 = 0.;
        double J03 = (std::cos(w*t)-1)/w;
        double J04 = 0.;
        double J05 = 0.;
        double J06 = (t*vx*std::cos(w*t)/w) - (t*vy*std::sin(w*t)/w) - (vx*std::sin(w*t)/std::pow(w,2)) - (vy*(std::cos(w*t)-1)/std::pow(w,2));

        double J10 = 0.;
        double J11 = std::cos(w*t);
        double J12 = 0.;
        double J13 = -std::sin(w*t);
        double J14 = 0.;
        double J15 = 0.;
        double J16 = -t*vx*std::sin(w*t) - t*vy*std::cos(w*t);

        double J20 = 0.;
        double J21 = (1-std::cos(w*t))/w;
        double J22 = 1.;
        double J23 = std::sin(w*t)/w;
        double J24 = 0.;
        double J25 = 0.;
        double J26 = (t*vx*std::sin(w*t)/w) + (t*vy*std::cos(w*t)/w) - (vx*(1-std::cos(w*t))/std::pow(w,2)) - (vy*std::sin(w*t)/std::pow(w,2));

        double J30 = 0.;
        double J31 = std::sin(w*t);
        double J32 = 0.;
        double J33 = std::cos(w*t);
        double J34 = 0.;
        double J35 = 0.;
        double J36 = t*vx*std::cos(w*t) - t*vy*std::sin(w*t);

        double J40 = 0.;
        double J41 = 0.;
        double J42 = 0.;
        double J43 = 0.;
        double J44 = 1.;
        double J45 = t;
        double J46 = 0.;

        double J50 = 0.;
        double J51 = 0.;
        double J52 = 0.;
        double J53 = 0.;
        double J54 = 0.;
        double J55 = 1.;
        double J56 = 0.;

        double J60 = 0.;
        double J61 = 0.;
        double J62 = 0.;
        double J63 = 0.;
        double J64 = 0.;
        double J65 = 0.;
        double J66 = 1.;

        J << J00, J01, J02, J03, J04, J05, J06,
             J10, J11, J12, J13, J14, J15, J16,
             J20, J21, J22, J23, J24, J25, J26,
             J30, J31, J32, J33, J34, J35, J36,
             J40, J41, J42, J43, J44, J45, J46,
             J50, J51, J52, J53, J54, J55, J56,
             J60, J61, J62, J63, J64, J65, J66;

        return J;
    }
};
//-----------------------------------------------------------------
template <class M>
struct G
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
template <class M>
struct H
{
    M operator()(const M& x, const M& z = M{})
    {
        M h(3,7);
        h << 1., 0., 0., 0., 0., 0., 0.,
             0., 0., 1., 0., 0., 0., 0.,
             0., 0., 0., 0., 1., 0., 0.;
        return h*x;
    }
    M operator()()
    {
        M h(3,7);
        h << 1., 0., 0., 0., 0., 0., 0.,
             0., 0., 1., 0., 0., 0., 0.,
             0., 0., 0., 0., 1., 0., 0.;
        return h;
    }
};
//-----------------------------------------------------------------
template <class M>
struct H_Jacobian
{
    M operator()(const M& x, const M& z = M{})
    {
        M h(3,7);
        h << 1., 0., 0., 0., 0., 0., 0.,
             0., 0., 1., 0., 0., 0., 0.,
             0., 0., 0., 0., 1., 0., 0.;
        return h*x;
    }
    M operator()()
    {
        M h(3,7);
        h << 1., 0., 0., 0., 0., 0., 0.,
             0., 0., 1., 0., 0., 0., 0.,
             0., 0., 0., 0., 1., 0., 0.;
        return h;
    }
};
//-----------------------------------------------------------------
}

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

        if(T!=6)
            std::cout << "T:" << T << std::endl;

        //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        //std::cout << "++++++++++++++++++++++++++++++" << std::endl;
        //std::cout << "CT-w:" << w << std::endl;
        double wt = w*T;
        //std::cout << "wt:" << wt << std::endl;
        if(w>=M_PI)
        {
            double wt_new = wt - 2*M_PI;
            double sin_wt = std::sin(wt);
            double sin_wt_new = std::sin(wt_new);
            double cos_wt = std::cos(wt);
            double cos_wt_new = std::cos(wt_new);
            std::cout << "CT[-]: " << w << std::endl;
            if(fabs(sin_wt - sin_wt_new) > 0.00000001)
                std::cout << "delta SIN: " << fabs(sin_wt - sin_wt_new) << std::endl;
            if(fabs(cos_wt - cos_wt_new) > 0.00000001)
                std::cout << "delta COS: " << fabs(cos_wt - cos_wt_new) << std::endl;
            if(sin_wt/sin_wt_new < 0 || cos_wt/cos_wt_new < 0)
                std::cout << "XXX" << std::endl;
        }
        if(w<=-M_PI)
        {
            double wt_new = wt + 2*M_PI;
            double sin_wt = std::sin(wt);
            double sin_wt_new = std::sin(wt_new);
            double cos_wt = std::cos(wt);
            double cos_wt_new = std::cos(wt_new);
            std::cout << "CT[+]: " << w << std::endl;
            if(fabs(sin_wt - sin_wt_new) > 0.00000001)
                std::cout << "delta SIN: " << fabs(sin_wt - sin_wt_new) << std::endl;
            if(fabs(cos_wt - cos_wt_new) > 0.00000001)
                std::cout << "delta COS: " << fabs(cos_wt - cos_wt_new) << std::endl;
            if(sin_wt/sin_wt_new < 0 || cos_wt/cos_wt_new < 0)
                std::cout << "XXX" << std::endl;
        }
        //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
//-----------------------------------------------------------------
template <class M>
struct FJacobian_CV
{
    enum class POSITION{X=0,VX=1,Y=2,VY=3,Z=4,VZ=5};
    M operator()(double T)
    {
        M J(6,6);
        J.setZero();
        J = {{1,T,0,0,0,0},
             {0,1,0,0,0,0},
             {0,0,1,T,0,0},
             {0,0,0,1,0,0},
             {0,0,0,0,1,T},
             {0,0,0,0,0,1}};
        return J;
    }
};
//-----------------------------------------------------------------
template <class M>
struct FJacobian_CT
{
    enum class POSITION{X=0,VX=1,Y=2,VY=3,Z=4,VZ=5,W=6};
    M operator()(const M& xx,double T)
    {
        double x = xx(static_cast<int>(POSITION::X));
        double vx = xx(static_cast<int>(POSITION::VX));
        double y = xx(static_cast<int>(POSITION::Y));
        double vy = xx(static_cast<int>(POSITION::VY));
        double z = xx(static_cast<int>(POSITION::Z));
        double vz = xx(static_cast<int>(POSITION::VZ));
        double w = xx(static_cast<int>(POSITION::W));
        double t = T;

        if(w==0)
            w=Utils::eps();

        M J(7,7);
        J.setZero();

        double J00 = 1.;
        double J01 = std::sin(w*t)/w;
        double J02 = 0.;
        double J03 = (std::cos(w*t)-1)/w;
        double J04 = 0.;
        double J05 = 0.;
        double J06 = (t*vx*std::cos(w*t)/w) - (t*vy*std::sin(w*t)/w) - (vx*std::sin(w*t)/std::pow(w,2)) - (vy*(std::cos(w*t)-1)/std::pow(w,2));

        double J10 = 0.;
        double J11 = std::cos(w*t);
        double J12 = 0.;
        double J13 = -std::sin(w*t);
        double J14 = 0.;
        double J15 = 0.;
        double J16 = -t*vx*std::sin(w*t) - t*vy*std::cos(w*t);

        double J20 = 0.;
        double J21 = (1-std::cos(w*t))/w;
        double J22 = 1.;
        double J23 = std::sin(w*t)/w;
        double J24 = 0.;
        double J25 = 0.;
        double J26 = (t*vx*std::sin(w*t)/w) + (t*vy*std::cos(w*t)/w) - (vx*(1-std::cos(w*t))/std::pow(w,2)) - (vy*std::sin(w*t)/std::pow(w,2));

        double J30 = 0.;
        double J31 = std::sin(w*t);
        double J32 = 0.;
        double J33 = std::cos(w*t);
        double J34 = 0.;
        double J35 = 0.;
        double J36 = t*vx*std::cos(w*t) - t*vy*std::sin(w*t);

        double J40 = 0.;
        double J41 = 0.;
        double J42 = 0.;
        double J43 = 0.;
        double J44 = 1.;
        double J45 = t;
        double J46 = 0.;

        double J50 = 0.;
        double J51 = 0.;
        double J52 = 0.;
        double J53 = 0.;
        double J54 = 0.;
        double J55 = 1.;
        double J56 = 0.;

        double J60 = 0.;
        double J61 = 0.;
        double J62 = 0.;
        double J63 = 0.;
        double J64 = 0.;
        double J65 = 0.;
        double J66 = 1.;

        J << J00, J01, J02, J03, J04, J05, J06,
             J10, J11, J12, J13, J14, J15, J16,
             J20, J21, J22, J23, J24, J25, J26,
             J30, J31, J32, J33, J34, J35, J36,
             J40, J41, J42, J43, J44, J45, J46,
             J50, J51, J52, J53, J54, J55, J56,
             J60, J61, J62, J63, J64, J65, J66;

        return J;
    }
};
//-----------------------------------------------------------------
template<class M>
struct HJacobian_X_R
{
    enum class POSITION_X{X=0,VX=1,Y=2,VY=3,Z=4,VZ=5};
    enum class POSITION_Z{R=0,A=1,E=2};
    M operator()(const M& xx)
    {
        double x = xx(static_cast<int>(POSITION_X::X));
        double y = xx(static_cast<int>(POSITION_X::Y));
        double z = xx(static_cast<int>(POSITION_X::Z));

        double XYZ = std::pow(x,2)+std::pow(y,2)+std::pow(z,2);
        double sqrtXYZ = std::sqrt(XYZ);
        double XY = std::pow(x,2)+std::pow(y,2);
        double sqrtXY = std::sqrt(XY);
        M J = {{-x*z/(sqrtXY*XYZ),0,-y*z/(sqrtXY*XYZ),0,sqrtXY/XYZ,0},
               {-y/XY,0,x/XY,0,0,0},
               {x/sqrtXYZ,0,y/sqrtXYZ,0,z/sqrtXYZ,0}};
        return J;
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

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//std::cout << "++++++++++++++++++++++++++++++" << std::endl;
//std::cout << "J-w:" << w << std::endl;
//        double wt = w*t;
//        //std::cout << "wt:" << wt << std::endl;
//        if(w>=M_PI)
//        {
//            double wt_new = wt - 2*M_PI;
//            double sin_wt = std::sin(wt);
//            double sin_wt_new = std::sin(wt_new);
//            double cos_wt = std::cos(wt);
//            double cos_wt_new = std::cos(wt_new);
//            std::cout << "J[-]: " << w << std::endl;
//            if(fabs(sin_wt - sin_wt_new) > 0.00000001)
//                std::cout << "delta SIN: " << fabs(sin_wt - sin_wt_new) << std::endl;
//            if(fabs(cos_wt - cos_wt_new) > 0.00000001)
//                std::cout << "delta COS: " << fabs(cos_wt - cos_wt_new) << std::endl;
//            if(sin_wt/sin_wt_new < 0 || cos_wt/cos_wt_new < 0)
//                std::cout << "XXX" << std::endl;
//        }
//        if(w<=-M_PI)
//        {
//            double wt_new = wt + 2*M_PI;
//            double sin_wt = std::sin(wt);
//            double sin_wt_new = std::sin(wt_new);
//            double cos_wt = std::cos(wt);
//            double cos_wt_new = std::cos(wt_new);
//            std::cout << "J[+]: " << w << std::endl;
//            if(fabs(sin_wt - sin_wt_new) > 0.00000001)
//                std::cout << "delta SIN: " << fabs(sin_wt - sin_wt_new) << std::endl;
//            if(fabs(cos_wt - cos_wt_new) > 0.00000001)
//                std::cout << "delta COS: " << fabs(cos_wt - cos_wt_new) << std::endl;
//            if(sin_wt/sin_wt_new < 0 || cos_wt/cos_wt_new < 0)
//                std::cout << "XXX" << std::endl;
//        }
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
