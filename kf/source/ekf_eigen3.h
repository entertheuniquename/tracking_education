#pragma once

#include "utils.h"

#include <armadillo>
#include <functional>
#include <numeric>

#include <math.h>

namespace Estimator
{
template <class M = Eigen::MatrixXd>
struct ExtendedKalmanFilterMath
{
    struct Prediction
    {
        M x;
        M S;
        M dFdx;
    };

    template<class TypeFuncStateTransition,
             class TypeFuncTransitionJacobian,
             class ...TypeParam>
    Prediction predict(const M& Qs,
                       const M& x,
                       const M& S,
                       TypeFuncStateTransition f,
                       TypeFuncTransitionJacobian df,
                       TypeParam ...p
                       )
    {
        Prediction ans;
        std::cout << "Qs:" << std::endl << Qs << std::endl;
        std::cout << "x:" << std::endl << x << std::endl;
        std::cout << "S:" << std::endl << S << std::endl;


        //std::pair<M, M> predMatrix = jacobianMatrices_analitic_CT_Deg(Qs, x, f, df, p...);
        std::pair<M, M> predMatrix = jacobianConstTurn(x, p...);
        std::cout << "1234444444444" << std::endl;
        M Qsqrt = predMatrix.second * Qs;
        ans.x = f(x, p...);
        std::cout << "456" << std::endl;


        ans.S = Utils::qrFactor_A(predMatrix.first, S, Qsqrt);
        ans.dFdx = predMatrix.second;
        std::cout << "789" << std::endl;

        return ans;
    }

    template<class TypeMeasurementTransition,
             class TypeFuncTransitionJacobian,
             class ...TypeParam>
    std::pair<M,M> correct(const M& z, const M& Rs, const M& x, const M& S,
                           TypeMeasurementTransition h,
                           TypeFuncTransitionJacobian dh,
                           TypeParam ...p)
    {
        MeasurementJacobianAndCovariance r =
            getMeasurementJacobianAndCovariance(Rs, x, S, h, dh, p...);
        M residue = z - r.zEstimated;
        return correctStateAndSqrtCovariance(x, S, residue, r.Pxy, r.Sy, r.dHdx, r.Rsqrt);
    }

private:

    template<class TypeFuncStateTransition,
             class ...TypeParam>
    M numericJacobianAdditive(TypeFuncStateTransition func,
                              const M& x,
                              TypeParam ...p)
    {
        double relativeStep = sqrt(Utils::eps());
        double delta = relativeStep;
        M z = func(x, p...);
        size_t n = Utils::length(x);
        size_t m = Utils::length(z);
        M jacobian = Utils::zeros(m, n);
        for (size_t j=0; j<n; ++j)
        {
            M imvec = x;
            auto epsilon = std::max(delta, delta*std::abs(imvec(j)));
            imvec(j) = imvec(j) + epsilon;
            M imz = func(imvec, p...);
            M deltaz = imz-z;
            jacobian.col(j) = deltaz / epsilon;
        }
        return jacobian;
    }

    template<class TypeFuncStateTransition,
             class ...TypeParam>
    M numericJacobianAdditive02(TypeFuncStateTransition func,
                              const M& x,
                              TypeParam ...p)
    {
        double relativeStep = sqrt(Utils::eps());
        double delta = relativeStep;
        M z = func(x, p...);
        size_t n = Utils::length(x);
        size_t m = Utils::length(z);
        M jacobian = Utils::zeros(m, n);
        for (size_t j=0; j<n; ++j)
        {
            M imvec = z;
            auto epsilon = std::max(delta, delta*std::abs(imvec(j)));
            imvec(j) = imvec(j) + epsilon;
            M imz = func(imvec, 0.00001);
            M deltaz = imz-z;
            jacobian.col(j) = deltaz / epsilon;
        }
        return jacobian;
    }

    template<class TypeFuncTransition,
             class TypeFuncTransitionJacobian,
             class ...TypeParam>
    std::pair<M, M> jacobianMatrices(const M& Ps, const M& x,
                 TypeFuncTransition f,
                 TypeFuncTransitionJacobian df,
                 TypeParam ...p)
    {
        M J = df(x, p...);
        return std::make_pair(Ps, J);
    }

    template<class TypeFuncStateTransition,
             class ...TypeParam>
    std::pair<M, M> jacobianMatrices(const M& Ps, const M& x,
                 TypeFuncStateTransition f,
                 std::nullptr_t df,
                 TypeParam ...p)
    {
        M J = numericJacobianAdditive(f, x, p...);
        return std::make_pair(Ps, J);
    }

    template<class TypeFuncStateTransition,
             class ...TypeParam>
    std::pair<M, M> jacobianMatrices02(const M& Ps, const M& x,
                 TypeFuncStateTransition f,
                 std::nullptr_t df,
                 TypeParam ...p)
    {
        M J = numericJacobianAdditive02(f, x, p...);
        return std::make_pair(Ps, J);
    }

    double GT(double t){return t;}
    template<class TypeFuncStateTransition,
             class ...TypeParam>
    std::pair<M, M> jacobianMatrices_analitic_CT(const M& Ps, const M& xx,
                 TypeFuncStateTransition f,
                 std::nullptr_t df,
                 TypeParam ...p)
    {
        enum class POSITION{X=0,VX=1,Y=2,VY=3,Z=4,VZ=5,W=6};
        double x = xx(static_cast<int>(POSITION::X));
        double vx = xx(static_cast<int>(POSITION::VX));
        double y = xx(static_cast<int>(POSITION::Y));
        double vy = xx(static_cast<int>(POSITION::VY));
        double z = xx(static_cast<int>(POSITION::Z));
        double vz = xx(static_cast<int>(POSITION::VZ));
        double w = xx(static_cast<int>(POSITION::W));
        double t = GT(p...);
        double dt = t;
        double dt2 = pow(t, 2.);

        if(w==0)
            w=Utils::eps();

        //std::cout << "jacobianMatrices_analitic_CT::t: " << t << std::endl;

        M J(7,7);
        J.setZero();

        double J00 = 1.;
        double J01 = sin(w*t)/w;
        double J02 = 0.;
        double J03 = (cos(w*t)-1)/w;
        double J04 = 0.;
        double J05 = 0.;
        double J06 = ((t*vx*cos(w*t)/w) - (t*vy*sin(w*t)/w) - (vx*sin(w*t)/pow(w,2)) - (vy*(cos(w*t)-1)/pow(w,2)));

        double J10 = 0.;
        double J11 = cos(w*t);
        double J12 = 0.;
        double J13 = -sin(w*t);
        double J14 = 0.;
        double J15 = 0.;
        double J16 = (-t*vx*sin(w*t) - t*vy*cos(w*t));

        double J20 = 0.;
        double J21 = (1-cos(w*t))/w;
        double J22 = 1.;
        double J23 = sin(w*t)/w;
        double J24 = 0.;
        double J25 = 0.;
        double J26 = ((t*vx*sin(w*t)/w) + (t*vy*cos(w*t)/w) - (vx*(1-cos(w*t))/pow(w,2)) - (vy*sin(w*t)/pow(w,2)));

        double J30 = 0.;
        double J31 = sin(w*t);
        double J32 = 0.;
        double J33 = cos(w*t);
        double J34 = 0.;
        double J35 = 0.;
        double J36 = (t*vx*cos(w*t) - t*vy*sin(w*t));

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


        int L = 7;
        M dfdw = Utils::zeros(7, 4);

        size_t pnx  = 0,
               pny  = 1,
               pno  = 3,
               pnz  = 2;

        dfdw(0, pnx) = dt2;
        dfdw(1, pnx) = dt;
        dfdw(2, pny) = dt2;
        dfdw(3, pny) = dt;
        dfdw(6, pno) = dt;
        dfdw(4, pnz) = dt2;
        dfdw(5, pnz) = dt;


        std::cout << dfdw << std::endl;
        std::cout << Ps << std::endl;

        M Qsqrt = dfdw * Ps;

        std::cout << Qsqrt << std::endl;


        return std::make_pair(Qsqrt, J);
    }


    //template <class M>
    std::pair<M, M> jacobianConstTurn(const M& state,
                                      double dt) {

        enum class POSITION{X=0,VX=1,Y=2,VY=3,Z=4,VZ=5,W=6};


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


        int L = 4;
        M dfdw = Utils::zeros(K, L);

        size_t pnx  = 0,
               pny  = 1,
               pnz  = 2,
               pno  = 3;

        dfdw(px , pnx) = dt2;
        dfdw(pvx, pnx) = dt;
        dfdw(py , pny) = dt2;
        dfdw(pvy, pny) = dt;
        dfdw(pz , pnz) = dt2;
        dfdw(pvz, pnz) = dt;
        dfdw(po , pno) = dt;

        return std::make_pair(jac, dfdw);
    }

    template<class TypeFuncStateTransition,
             class ...TypeParam>
    std::pair<M, M> jacobianMatrices_analitic_CT_Deg(const M& Ps, const M& xx,
                 TypeFuncStateTransition f,
                 std::nullptr_t df,
                 TypeParam ...p)
    {

        std::cout << "xx=" << xx << std::endl;
        enum class POSITION{X=0,VX=1,Y=2,VY=3,Z=4,VZ=5,W=6};
        double x = xx(static_cast<int>(POSITION::X));
        double vx = xx(static_cast<int>(POSITION::VX));
        double y = xx(static_cast<int>(POSITION::Y));
        double vy = xx(static_cast<int>(POSITION::VY));
        double z = xx(static_cast<int>(POSITION::Z));
        double vz = xx(static_cast<int>(POSITION::VZ));
        double w = xx(static_cast<int>(POSITION::W));

        w = Utils::deg2rad(w);

        double t = GT(p...);
        double dt = t;
        double dt2 = pow(t, 2.) / 2.;

        if(w==0)
            w=Utils::eps();

        //std::cout << "jacobianMatrices_analitic_CT::t: " << t << std::endl;

        M J(7,7);
        J.setZero();



        double J00 = 1.;
        double J01 = sin(w*t)/w;
        double J02 = 0.;
        double J03 = (cos(w*t)-1)/w;
        double J04 = 0.;
        double J05 = 0.;
        double J06 = Utils::deg2rad((t*vx*cos(w*t)/w) - (t*vy*sin(w*t)/w) - (vx*sin(w*t)/pow(w,2)) - (vy*(cos(w*t)-1)/pow(w,2)));

        double J10 = 0.;
        double J11 = cos(w*t);
        double J12 = 0.;
        double J13 = -sin(w*t);
        double J14 = 0.;
        double J15 = 0.;
        double J16 = Utils::deg2rad(-t*vx*sin(w*t) - t*vy*cos(w*t));

        double J20 = 0.;
        double J21 = (1-cos(w*t))/w;
        double J22 = 1.;
        double J23 = sin(w*t)/w;
        double J24 = 0.;
        double J25 = 0.;
        double J26 = Utils::deg2rad((t*vx*sin(w*t)/w) + (t*vy*cos(w*t)/w) - (vx*(1-cos(w*t))/pow(w,2)) - (vy*sin(w*t)/pow(w,2)));

        double J30 = 0.;
        double J31 = sin(w*t);
        double J32 = 0.;
        double J33 = cos(w*t);
        double J34 = 0.;
        double J35 = 0.;
        double J36 = Utils::deg2rad(t*vx*cos(w*t) - t*vy*sin(w*t));

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


        int L = 7;
        M dfdw = Utils::zeros(7, 4);

        size_t pnx  = 0,
               pny  = 1,
               pno  = 3,
               pnz  = 2;

        dfdw(0, pnx) = dt2;
        dfdw(1, pnx) = dt;
        dfdw(2, pny) = dt2;
        dfdw(3, pny) = dt;
        dfdw(6, pno) = dt;
        dfdw(4, pnz) = dt2;
        dfdw(5, pnz) = dt;


        std::cout << dfdw << std::endl;
        std::cout << Ps << std::endl;

        M Qsqrt = dfdw * Ps;

        std::cout << Qsqrt << std::endl;

        return std::make_pair(Qsqrt, J);
    }

    template<class TypeFuncStateTransition,
             class ...TypeParam>
    std::pair<M, M> jacobianMatrices3A_Analitic_Predict(const M& Ps, const M& x,
                 TypeFuncStateTransition f,
                 std::nullptr_t df,
                 TypeParam ...p)
    {
        M J = {{1,0.2,0,0,0,0},
               {0,1,0,0,0,0},
               {0,0,1,0.2,0,0},
               {0,0,0,1,0,0},
               {0,0,0,0,1,0.2},
               {0,0,0,0,0,1}};
        return std::make_pair(Ps, J);
    }

    template<class TypeFuncStateTransition,
             class ...TypeParam>
    std::pair<M, M> jacobianMatrices3A_Analitic_Correct(const M& Ps, const M& x,
                 TypeFuncStateTransition f,
                 std::nullptr_t df,
                 TypeParam ...p)
    {
        double X = x[0];
        double Y = x[2];
        double Z = x[4];
        double XYZ = x[0]*x[0]+x[2]*x[2]+x[4]*x[4];
        double sqrtXYZ = std::sqrt(XYZ);
        double XY = x[0]*x[0]+x[2]*x[2];
        double sqrtXY = std::sqrt(XY);
        M J = {{-X*Z/(sqrtXY*XYZ),0,-Y*Z/(sqrtXY*XYZ),0,sqrtXY/XYZ,0},
               {-Y/XY,0,X/XY,0,0,0},
               {X/sqrtXYZ,0,Y/sqrtXYZ,0,Z/sqrtXYZ,0}};
        return std::make_pair(Ps, J);
    }

    struct MeasurementJacobianAndCovariance
    {
        M zEstimated;
        M Pxy;
        M Sy;
        M dHdx;
        M Rsqrt;
    };

    template<class TypeMeasurementTransition,
             class TypeFuncTransitionJacobian,
             class ...TypeParam>
    MeasurementJacobianAndCovariance getMeasurementJacobianAndCovariance(
                                        const M& Rs, const M& x, const M& S,
                                        TypeMeasurementTransition h,
                                        TypeFuncTransitionJacobian dh,
                                        TypeParam ...p)
    {
        MeasurementJacobianAndCovariance ans;
        std::pair<M, M> measMatrix = jacobianMatrices(Rs, x, h, dh, p...);
        ans.Rsqrt = measMatrix.first;
        ans.dHdx = measMatrix.second;
        std::cout << "dHdx=" << ans.dHdx << std::endl;
        ans.zEstimated = h(x, p...);
        ans.Pxy = (S*Utils::transpose(S))*Utils::transpose(ans.dHdx);
        ans.Sy = Utils::qrFactor_A(ans.dHdx, S, ans.Rsqrt);
        return ans;
    }

    template<class TypeMeasurementTransition,
             class TypeFuncTransitionJacobian,
             class ...TypeParam>
    MeasurementJacobianAndCovariance getMeasurementJacobianAndCovariance3AA(
                                        const M& Rs, const M& x, const M& S,
                                        TypeMeasurementTransition h,
                                        TypeFuncTransitionJacobian dh,
                                        TypeParam ...p)
    {
        MeasurementJacobianAndCovariance ans;
        std::pair<M, M> measMatrix = jacobianMatrices3A_Analitic_Correct(Rs, x, h, dh, p...);

        ans.Rsqrt = measMatrix.first;
        ans.dHdx = measMatrix.second;
        ans.zEstimated = h(x, p...);
        ans.Pxy = (S*trans(S))*trans(ans.dHdx);
        ans.Sy = Utils::qrFactor_A(ans.dHdx, S, ans.Rsqrt);

        return ans;
    }

    std::pair<M,M> correctStateAndSqrtCovariance(const M& x,
                                                 const M& S,
                                                 const M& residue,
                                                 const M& Pxy,
                                                 const M& Sy,
                                                 const M& H,
                                                 const M& Rsqrt)
    {
        M K = Utils::ComputeKalmanGain_A(Sy, Pxy);
        M xp = x + K * residue;
        M A  = -K * H;
        for (int i=0; i<A.rows(); ++i) {
            A(i, i) = 1. + A(i, i);
        }
        M Ks = K*Rsqrt;
        M Sp = Utils::qrFactor_A(A, S, Ks);

        return std::make_pair(xp, Sp);
    }
};

template<class M,class SM, class MM>
struct EKFE : public ExtendedKalmanFilterMath<M>
{
private:
    Eigen::MatrixXd sqrtStateCovariance;
    Eigen::MatrixXd sqrtProcessNoise;
    Eigen::MatrixXd sqrtMeasurementNoise;
    Eigen::MatrixXd State;
public:
    EKFE(Eigen::MatrixXd state,
         Eigen::MatrixXd covariance,
         Eigen::MatrixXd processNoise,
         Eigen::MatrixXd measureNoise):
        State(state)
    {
        std::cout << "P0=" << std::endl
                  << covariance << std::endl;

        SetStateCovariance(covariance);
        SetProcessNoise(processNoise);
        SetMeasurementNoise(measureNoise);
    }

    EKFE& SetStateCovariance(const Eigen::MatrixXd& StateCovariance)
    {
        ////std::cout << "SetStateCovariance" << std::endl;
        CHECK_SYMETRIC_POSITIVE(Utils::EA(StateCovariance));
        sqrtStateCovariance = Utils::cholPSD_A(StateCovariance);
        return *this;
    }

    EKFE& SetProcessNoise(const Eigen::MatrixXd& ProcessNoise)
    {
        std::cout << ProcessNoise << std::endl;

        ////std::cout << "SetProcessNoise" << std::endl;
        CHECK_SYMETRIC_POSITIVE(Utils::EA(ProcessNoise));

        arma::mat ProcessNoiseA = Utils::EA(ProcessNoise);
        std::cout << ProcessNoiseA << std::endl;


        sqrtProcessNoise = Utils::cholPSD_A(ProcessNoise);
        std::cout << sqrtProcessNoise << std::endl;
        return *this;
    }

    EKFE& SetMeasurementNoise(const Eigen::MatrixXd& MeasurementNoise)
    {
        ////std::cout << "SetMeasurementNoise" << std::endl;
        CHECK_SYMETRIC_POSITIVE(Utils::EA(MeasurementNoise));
        sqrtMeasurementNoise = Utils::cholPSD_A(MeasurementNoise);
        return *this;
    }

    Eigen::MatrixXd GetStateCovariance() const
    {
        return sqrtStateCovariance * Utils::transpose(sqrtStateCovariance);
    }

    Eigen::MatrixXd GetProcessNoise() const
    {
        return sqrtProcessNoise * Utils::transpose(sqrtProcessNoise);
    }

    Eigen::MatrixXd GetMeasurementNoise() const
    {
        return sqrtMeasurementNoise * Utils::transpose(sqrtMeasurementNoise);
    }

    Eigen::MatrixXd GetState() const
    {
        return State;
    }

    template <class ...TypeParam>
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> predict(double dt,TypeParam ...param)
    {
        std::cout << "predict" << std::endl;
        auto pred = ExtendedKalmanFilterMath<>::predict(sqrtProcessNoise,
                                                        State,
                                                        sqrtStateCovariance,
                                                        SM(),
                                                        nullptr,
                                                        dt);
        State = pred.x;
        sqrtStateCovariance = pred.S;
        return std::make_pair(State, GetStateCovariance());
    }

    template <class ...TypeParam>
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> correct(const Eigen::MatrixXd& measurement,
                                                            TypeParam ...param)
    {
        std::cout << "correct" << std::endl;
        auto corr = ExtendedKalmanFilterMath<>::correct(measurement,
                                                        sqrtMeasurementNoise,
                                                        State,
                                                        sqrtStateCovariance,
                                                        MM(),
                                                        nullptr,
                                                        measurement//#TODO! - сделать через параметр
                                                        /*param...*/);//#TODO!
        State = corr.first;
        sqrtStateCovariance = corr.second;
        return std::make_pair(State, GetStateCovariance());
    }

};

}





