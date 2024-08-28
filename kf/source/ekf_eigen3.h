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
        std::pair<M, M> predMatrix = jacobianConstTurn(x, p...);
        M Qsqrt = predMatrix.second * Qs;
        ans.x = f(x, p...);

        ans.S = Utils::qrFactor_A(predMatrix.first, S, Qsqrt);
        ans.dFdx = predMatrix.second;

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
        ans.zEstimated = h(x, p...);
        ans.Pxy = (S*Utils::transpose(S))*Utils::transpose(ans.dHdx);
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
        SetStateCovariance(covariance);
        SetProcessNoise(processNoise);
        SetMeasurementNoise(measureNoise);
    }

    EKFE& SetStateCovariance(const Eigen::MatrixXd& StateCovariance)
    {
        CHECK_SYMETRIC_POSITIVE(Utils::EA(StateCovariance));
        sqrtStateCovariance = Utils::cholPSD_A(StateCovariance);
        return *this;
    }

    EKFE& SetProcessNoise(const Eigen::MatrixXd& ProcessNoise)
    {
        CHECK_SYMETRIC_POSITIVE(Utils::EA(ProcessNoise));
        sqrtProcessNoise = Utils::cholPSD_A(ProcessNoise);
        return *this;
    }

    EKFE& SetMeasurementNoise(const Eigen::MatrixXd& MeasurementNoise)
    {
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





