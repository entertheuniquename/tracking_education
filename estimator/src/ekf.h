#pragma once

#include "utils.h"

#include <armadillo>
#include <functional>
#include <numeric>

namespace Estimator {

template <class M = arma::Mat<double>>
struct ExtendedKalmanFilterMath
{    
    struct Prediction {
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
                       ) {
        Prediction ans;
        std::pair<M, M> predMatrix = jacobianMatrices(Qs, x, f, df, p...);
        ans.x = f(x, p...);
        ans.S = Utils::qrFactor(predMatrix.second, S, predMatrix.first);
        ans.dFdx = predMatrix.second;
        return ans;
    }

    template<class TypeFuncStateTransition,
             class TypeFuncTransitionJacobian,
             class ...TypeParam>
    Prediction predict3AA(const M& Qs,
                          const M& x,
                          const M& S,
                          TypeFuncStateTransition f,
                          TypeFuncTransitionJacobian df,
                          TypeParam ...p
                          ) {
        Prediction ans;
        std::pair<M, M> predMatrix = jacobianMatrices3A_Analitic_Predict(Qs, x, f, df, p...);
        ans.x = f(x, p...);
        ans.S = Utils::qrFactor(predMatrix.second, S, predMatrix.first);
        ans.dFdx = predMatrix.second;
        return ans;
    }

    template<class TypeMeasurementTransition,
             class TypeFuncTransitionJacobian,
             class ...TypeParam>
    std::pair<M,M> correct(const M& z, const M& Rs, const M& x, const M& S,
                           TypeMeasurementTransition h,
                           TypeFuncTransitionJacobian dh,
                           TypeParam ...p) {
        MeasurementJacobianAndCovariance r =
            getMeasurementJacobianAndCovariance(Rs, x, S, h, dh, p...);
        M residue = z - r.zEstimated;
        return correctStateAndSqrtCovariance(x, S, residue, r.Pxy, r.Sy, r.dHdx, r.Rsqrt);
    }

    template<class TypeMeasurementTransition,
             class TypeFuncTransitionJacobian,
             class ...TypeParam>
    std::pair<M,M> correct3AA(const M& z, const M& Rs, const M& x, const M& S,
                           TypeMeasurementTransition h,
                           TypeFuncTransitionJacobian dh,
                           TypeParam ...p) {
        MeasurementJacobianAndCovariance r =
            getMeasurementJacobianAndCovariance3AA(Rs, x, S, h, dh, p...);
        M residue = z - r.zEstimated;
        return correctStateAndSqrtCovariance(x, S, residue, r.Pxy, r.Sy, r.dHdx, r.Rsqrt);
    }

private:

    template<class TypeFuncStateTransition,
             class ...TypeParam>
    M numericJacobianAdditive(TypeFuncStateTransition func,
                              const M& x,
                              TypeParam ...p) {

        double relativeStep = sqrt(Utils::eps());
        double delta = relativeStep;

        M z = func(x, p...);
        size_t n = Utils::length(x);
        size_t m = Utils::length(z);
        M jacobian = Utils::zeros(m, n);

        for (size_t j=0; j<n; ++j) {
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
                 TypeParam ...p) {
        M J = df(x, p...);
        return std::make_pair(Ps, J);
    }

    template<class TypeFuncStateTransition,
             class ...TypeParam>
    std::pair<M, M> jacobianMatrices(const M& Ps, const M& x,
                 TypeFuncStateTransition f,
                 std::nullptr_t df,
                 TypeParam ...p) {
        M J = numericJacobianAdditive(f, x, p...);
        return std::make_pair(Ps, J);
    }

    template<class TypeFuncStateTransition,
             class ...TypeParam>
    std::pair<M, M> jacobianMatrices3A_Analitic_Predict(const M& Ps, const M& x,
                 TypeFuncStateTransition f,
                 std::nullptr_t df,
                 TypeParam ...p) {
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
                 TypeParam ...p) {
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

    struct MeasurementJacobianAndCovariance {
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
                                        TypeParam ...p) {
        MeasurementJacobianAndCovariance ans;
        std::pair<M, M> measMatrix = jacobianMatrices(Rs, x, h, dh, p...);

        ans.Rsqrt = measMatrix.first;
        ans.dHdx = measMatrix.second;
        ans.zEstimated = h(x, p...);
        ans.Pxy = (S*trans(S))*trans(ans.dHdx);
        ans.Sy = Utils::qrFactor(ans.dHdx, S, ans.Rsqrt);

        return ans;
    }

    template<class TypeMeasurementTransition,
             class TypeFuncTransitionJacobian,
             class ...TypeParam>
    MeasurementJacobianAndCovariance getMeasurementJacobianAndCovariance3AA(
                                        const M& Rs, const M& x, const M& S,
                                        TypeMeasurementTransition h,
                                        TypeFuncTransitionJacobian dh,
                                        TypeParam ...p) {
        MeasurementJacobianAndCovariance ans;
        std::pair<M, M> measMatrix = jacobianMatrices3A_Analitic_Correct(Rs, x, h, dh, p...);

        ans.Rsqrt = measMatrix.first;
        ans.dHdx = measMatrix.second;
        ans.zEstimated = h(x, p...);
        ans.Pxy = (S*trans(S))*trans(ans.dHdx);
        ans.Sy = Utils::qrFactor(ans.dHdx, S, ans.Rsqrt);

        return ans;
    }

    std::pair<M,M> correctStateAndSqrtCovariance(const M& x,
                                                 const M& S,
                                                 const M& residue,
                                                 const M& Pxy,
                                                 const M& Sy,
                                                 const M& H,
                                                 const M& Rsqrt) {
        M K = Utils::ComputeKalmanGain(Sy, Pxy);
        M xp = x + K * residue;
        M A  = -K * H;
        for (int i=0; i<A.n_rows; ++i) {
            A(i, i) = 1. + A(i, i);
        }
        M Ks = K*Rsqrt;
        M Sp = Utils::qrFactor(A, S, Ks);

        return std::make_pair(xp, Sp);
    }
};

struct ExtendedKalmanFilter : public ExtendedKalmanFilterMath<> {

    arma::Mat<double> State;

    ExtendedKalmanFilter(arma::Mat<double> state,
                         arma::Mat<double> covariance,
                         arma::Mat<double> processNoise,
                         arma::Mat<double> measureNoise
                         ) : State(state)
    {
        SetStateCovariance(covariance);
        SetProcessNoise(processNoise);
        SetMeasurementNoise(measureNoise);
    }

    template <class TypeFuncStateTransition,
              class ...TypeParam>
    std::pair<arma::Mat<double>, arma::Mat<double>> predict(TypeFuncStateTransition stateModel,
                                                            TypeParam ...param) {
        auto pred = ExtendedKalmanFilterMath<>::predict(sqrtProcessNoise,
                                                        State,
                                                        sqrtStateCovariance,
                                                        stateModel,
                                                        nullptr,
                                                        param...
                                                        );
        State = pred.x;
        sqrtStateCovariance = pred.S;
        return std::make_pair(State, GetStateCovariance());
    }

    template <class TypeFuncStateTransition,
              class ...TypeParam>
    std::pair<arma::Mat<double>, arma::Mat<double>> predict3AA(TypeFuncStateTransition stateModel,
                                                            TypeParam ...param) {
        auto pred = ExtendedKalmanFilterMath<>::predict3AA(sqrtProcessNoise,
                                                           State,
                                                           sqrtStateCovariance,
                                                           stateModel,
                                                           nullptr,
                                                           param...
                                                           );
        State = pred.x;
        sqrtStateCovariance = pred.S;
        return std::make_pair(State, GetStateCovariance());
    }

    template <class TypeFuncMeasurementTransition,
              class ...TypeParam>
    std::pair<arma::Mat<double>, arma::Mat<double>> correct(const arma::mat& measurement,
                                                            TypeFuncMeasurementTransition measureModel,
                                                            TypeParam ...param)
    {
        auto corr = ExtendedKalmanFilterMath<>::correct(measurement,
                                                        sqrtMeasurementNoise,
                                                        State,
                                                        sqrtStateCovariance,
                                                        measureModel,
                                                        nullptr,
                                                        param...);


        State = corr.first;
        sqrtStateCovariance = corr.second;
        return std::make_pair(State, GetStateCovariance());
    }

    template <class TypeFuncMeasurementTransition,
              class ...TypeParam>
    std::pair<arma::Mat<double>, arma::Mat<double>> correct3AA(const arma::mat& measurement,
                                                            TypeFuncMeasurementTransition measureModel,
                                                            TypeParam ...param)
    {
        auto corr = ExtendedKalmanFilterMath<>::correct3AA(measurement,
                                                        sqrtMeasurementNoise,
                                                        State,
                                                        sqrtStateCovariance,
                                                        measureModel,
                                                        nullptr,
                                                        param...);


        State = corr.first;
        sqrtStateCovariance = corr.second;
        return std::make_pair(State, GetStateCovariance());
    }

    ExtendedKalmanFilter& SetStateCovariance(const arma::Mat<double>& StateCovariance) {
        CHECK_SYMETRIC_POSITIVE(StateCovariance);
        sqrtStateCovariance = Utils::cholPSD(StateCovariance);
        return *this;
    }

    ExtendedKalmanFilter& SetProcessNoise(const arma::Mat<double>& ProcessNoise) {
        CHECK_SYMETRIC_POSITIVE(ProcessNoise);
        sqrtProcessNoise = Utils::cholPSD(ProcessNoise);
        return *this;
    }

    ExtendedKalmanFilter& SetMeasurementNoise(const arma::Mat<double>& MeasurementNoise) {
        CHECK_SYMETRIC_POSITIVE(MeasurementNoise);
        sqrtMeasurementNoise = Utils::cholPSD(MeasurementNoise);
        return *this;
    }

    arma::Mat<double> GetStateCovariance() const {
        return sqrtStateCovariance * trans(sqrtStateCovariance);
    }

    arma::Mat<double> GetProcessNoise() const {
        return sqrtProcessNoise * trans(sqrtProcessNoise);
    }

    arma::Mat<double> GetMeasurementNoise() const {
        return sqrtMeasurementNoise * trans(sqrtMeasurementNoise);
    }

private:
    arma::Mat<double> sqrtStateCovariance;
    arma::Mat<double> sqrtProcessNoise;
    arma::Mat<double> sqrtMeasurementNoise;
};

}





