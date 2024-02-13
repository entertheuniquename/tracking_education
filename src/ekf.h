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
    Prediction predict(const M& Qs, //Q
                       const M& x, //x0
                       const M& S, //P0
                       TypeFuncStateTransition f, //stateModel
                       TypeFuncTransitionJacobian df, //.
                       TypeParam ...p //T
                       ) {
        //std::cout << ":::::::::: M ::::::::::" << std::endl;
        //std::cout << "Q:" << std::endl << Qs << std::endl;
        //std::cout << "x0:" << std::endl << x << std::endl;
        //std::cout << "P0:" << std::endl << S << std::endl;
        Prediction ans;
        std::pair<M, M> predMatrix = jacobianMatrices(Qs, x, f, df, p...);
        //return <Qs,J>
        //----------------------------------------------------------------
        //jacobian alternative

        //----------------------------------------------------------------
        ans.x = f(x, p...);
        //std::cout << "x:" << std::endl << ans.x << std::endl;
        ans.S = Utils::qrFactor(predMatrix.second, S, predMatrix.first);
        ans.dFdx = predMatrix.second; //J
        return ans;
    }

    template<class TypeFuncStateTransition,
             class TypeFuncTransitionJacobian,
             class ...TypeParam>
    Prediction predict3AA(const M& Qs, //Q
                          const M& x, //x0
                          const M& S, //P0
                          TypeFuncStateTransition f, //stateModel
                          TypeFuncTransitionJacobian df, //.
                          TypeParam ...p //T
                          ) {
        Prediction ans;
        std::pair<M, M> predMatrix = jacobianMatrices3A_Analitic_Predict(Qs, x, f, df, p...);
        //return <Qs,J>
        //----------------------------------------------------------------
        //jacobian alternative

        //----------------------------------------------------------------
        ans.x = f(x, p...);
        ans.S = Utils::qrFactor(predMatrix.second, S, predMatrix.first);
        ans.dFdx = predMatrix.second; //J
        return ans;
    }

    template<class TypeMeasurementTransition,
             class TypeFuncTransitionJacobian,
             class ...TypeParam>
    std::pair<M,M> correct(const M& z, const M& Rs, const M& x, const M& S,
                           TypeMeasurementTransition h, //measurementModel
                           TypeFuncTransitionJacobian dh,
                           TypeParam ...p) {
        //std::cout << "correct::z:" << z << std::endl;
        //std::cout << "correct::x:" << x << std::endl;
        MeasurementJacobianAndCovariance r =
            getMeasurementJacobianAndCovariance(Rs, x, S, h, dh, p...);
        //PRINTM(z);
        //PRINTM(r.zEstimated);
        //std::cout << "correct::r.zEstimated:" << r.zEstimated << std::endl;
        M residue = z - r.zEstimated;

        //PRINTM(residue);
        //std::cout << "correct::residue:" << residue << std::endl;
        return correctStateAndSqrtCovariance(x, S, residue, r.Pxy, r.Sy, r.dHdx, r.Rsqrt);
    }

    template<class TypeMeasurementTransition,
             class TypeFuncTransitionJacobian,
             class ...TypeParam>
    std::pair<M,M> correct3AA(const M& z, const M& Rs, const M& x, const M& S,
                           TypeMeasurementTransition h, //measurementModel
                           TypeFuncTransitionJacobian dh,
                           TypeParam ...p) {
        //std::cout << "correct::z:" << z << std::endl;
        //std::cout << "correct::x:" << x << std::endl;
        MeasurementJacobianAndCovariance r =
            getMeasurementJacobianAndCovariance3AA(Rs, x, S, h, dh, p...);
        //PRINTM(z);
        //PRINTM(r.zEstimated);
        //std::cout << "correct::r.zEstimated:" << r.zEstimated << std::endl;
        M residue = z - r.zEstimated;

        //PRINTM(residue);
        //std::cout << "correct::residue:" << residue << std::endl;
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
        //std::cout << "*** x:" << std::endl << x << std::endl;
        M J = numericJacobianAdditive(f, x, p...);
        //std::cout << "J:" << std::endl << J << std::endl;
        return std::make_pair(Ps, J);
    }

    template<class TypeFuncStateTransition,
             class ...TypeParam>
    std::pair<M, M> jacobianMatrices3A_Analitic_Predict(const M& Ps, const M& x,
                 TypeFuncStateTransition f,
                 std::nullptr_t df,
                 TypeParam ...p) {
        //=================================================
        //M J = {{x[0]/std::sqrt(x[0]*x[0]+x[2]*x[2]+x[4]*x[4]),-x[2],-x[4]*x[2]/std::sqrt(x[0]*x[0]+x[2]*x[2])},
        //       {x[2]/std::sqrt(x[0]*x[0]+x[2]*x[2]+x[4]*x[4]), x[0],-x[4]*x[0]/std::sqrt(x[0]*x[0]+x[2]*x[2])},
        //       {x[4]/std::sqrt(x[0]*x[0]+x[2]*x[2]+x[4]*x[4]), 0,std::sqrt(x[0]*x[0]+x[2]*x[2])}};
        //=================================================
        M J = {{1,0.2,0,0,0,0},
               {0,1,0,0,0,0},
               {0,0,1,0.2,0,0},
               {0,0,0,1,0,0},
               {0,0,0,0,1,0.2},
               {0,0,0,0,0,1}};
        //=================================================
        //std::cout << "J:" << std::endl << J << std::endl;
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
        //=[4]=============================================
        M J = {{-X*Z/(sqrtXY*XYZ),0,-Y*Z/(sqrtXY*XYZ),0,sqrtXY/XYZ,0},
               {-Y/XY,0,X/XY,0,0,0},
               {X/sqrtXYZ,0,Y/sqrtXYZ,0,Z/sqrtXYZ,0}};
        //=================================================
        //std::cout << "J:" << std::endl << J << std::endl;
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
        //std::cout << "K: " << K << std::endl;
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

    ExtendedKalmanFilter(arma::Mat<double> state, //x0
                         arma::Mat<double> covariance, //P0
                         arma::Mat<double> processNoise, //Q
                         arma::Mat<double> measureNoise //R
                         ) : State(state) //x0
    {
        SetStateCovariance(covariance); //sqrtStateCovariance //P0
        SetProcessNoise(processNoise); //sqrtProcessNoise //Q
        SetMeasurementNoise(measureNoise); //sqrtMeasurementNoise //R
    }

    template <class TypeFuncStateTransition,
              class ...TypeParam>
    std::pair<arma::Mat<double>, arma::Mat<double>> predict(TypeFuncStateTransition stateModel,
                                                            TypeParam ...param) {
        auto pred = ExtendedKalmanFilterMath<>::predict(sqrtProcessNoise, //Q
                                                        State, //x0
                                                        sqrtStateCovariance, //P0
                                                        stateModel, //stateModel <-
                                                        nullptr,
                                                        param... //T <-
                                                        );
        State = pred.x;
        sqrtStateCovariance = pred.S;
        return std::make_pair(State, GetStateCovariance());
    }

    template <class TypeFuncStateTransition,
              class ...TypeParam>
    std::pair<arma::Mat<double>, arma::Mat<double>> predict3AA(TypeFuncStateTransition stateModel,
                                                            TypeParam ...param) {
        auto pred = ExtendedKalmanFilterMath<>::predict3AA(sqrtProcessNoise, //Q
                                                           State, //x0
                                                           sqrtStateCovariance, //P0
                                                           stateModel, //stateModel <-
                                                           nullptr,
                                                           param... //T <-
                                                           );
        State = pred.x;
        sqrtStateCovariance = pred.S;
        return std::make_pair(State, GetStateCovariance());
    }

    template <class TypeFuncMeasurementTransition,
              class ...TypeParam>
    std::pair<arma::Mat<double>, arma::Mat<double>> correct(const arma::mat& measurement,//z
                                                            TypeFuncMeasurementTransition measureModel,
                                                            TypeParam ...param)//z
    {
        auto corr = ExtendedKalmanFilterMath<>::correct(measurement,
                                                        sqrtMeasurementNoise,
                                                        State,
                                                        sqrtStateCovariance,
                                                        measureModel,
                                                        nullptr,
                                                        param...);


        State = corr.first;

        //std::cout << "correct::State:" << State << std::endl;

        sqrtStateCovariance = corr.second;
        return std::make_pair(State, GetStateCovariance());
    }

    template <class TypeFuncMeasurementTransition,
              class ...TypeParam>
    std::pair<arma::Mat<double>, arma::Mat<double>> correct3AA(const arma::mat& measurement,//z
                                                            TypeFuncMeasurementTransition measureModel,
                                                            TypeParam ...param)//z
    {
        auto corr = ExtendedKalmanFilterMath<>::correct3AA(measurement,
                                                        sqrtMeasurementNoise,
                                                        State,
                                                        sqrtStateCovariance,
                                                        measureModel,
                                                        nullptr,
                                                        param...);


        State = corr.first;

        //std::cout << "correct::State:" << State << std::endl;

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





