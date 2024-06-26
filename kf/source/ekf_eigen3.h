#pragma once

#include "utils.h"

#include <armadillo>
#include <functional>
#include <numeric>

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
        std::pair<M, M> predMatrix = jacobianMatrices(Qs, x, f, df, p...);
        ans.x = f(x, p...);
        ans.S = Utils::qrFactor_A(predMatrix.second, S, predMatrix.first);
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
        SetStateCovariance(covariance);
        SetProcessNoise(processNoise);
        SetMeasurementNoise(measureNoise);
    }

    EKFE& SetStateCovariance(const Eigen::MatrixXd& StateCovariance)
    {
        sqrtStateCovariance = Utils::cholPSD_A(StateCovariance);
        return *this;
    }

    EKFE& SetProcessNoise(const Eigen::MatrixXd& ProcessNoise)
    {
        sqrtProcessNoise = Utils::cholPSD_A(ProcessNoise);
        return *this;
    }

    EKFE& SetMeasurementNoise(const Eigen::MatrixXd& MeasurementNoise)
    {
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





