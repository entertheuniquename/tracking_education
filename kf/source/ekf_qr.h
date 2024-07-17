#pragma once

#include "utils.h"
#include "filter.h"

namespace Estimator
{
template <class M>
struct EKF_QRMath
{
    struct Prediction
    {
        M x;
        M S;
        M dFdx;
    };

    template<class TypeFuncstateTransition,
             class TypeFuncTransitionJacobian,
             class ...TypeParam>
    Prediction predict(const M& Qs,
                       const M& x,
                       const M& S,
                       TypeFuncstateTransition f,
                       TypeFuncTransitionJacobian df,
                       TypeParam ...p
                       )
    {
        Prediction ans;
        M J = jacobianMatrices(x, f, df, p...);
        ans.x = f(x, p...);
        //ans.x = predMatrix.second*x;//#TEMP
        ans.S = Utils::qrFactor_A(J, S, Qs);
        ans.dFdx = J;
        return ans;
    }

    template<class TypeMeasurementTransition,
             class TypeFuncTransitionJacobian,
             class ...TypeParam>
    std::pair<M,M> correct(const M& z, M& zp, M& Se, M& dz, const M& Rs, const M& x, const M& S,
                           TypeMeasurementTransition h,
                           TypeFuncTransitionJacobian dh,
                           TypeParam ...p)
    {
        MeasurementJacobianAndCovariance r =
            getMeasurementJacobianAndCovariance(Rs, x, S, h, dh, p...);
        zp = r.zEstimated;
        dz = z - zp;
        Se = r.dHdx*(S*Utils::transpose(S))*Utils::transpose(r.dHdx) + Rs * Utils::transpose(Rs);//#TODO - ПРОВЕРИТЬ - делал на коленках. Похоже на правду.
        return correctstateAndSqrtCovariance(x, S, dz, r.Pxy, r.Sy, r.dHdx, r.Rsqrt);
    }

private:

    template<class TypeFuncstateTransition,
             class ...TypeParam>
    M numericJacobianAdditive(TypeFuncstateTransition func,
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
    M jacobianMatrices(const M& x,
                 TypeFuncTransition f,
                 TypeFuncTransitionJacobian df,
                 TypeParam ...p)
    {
        M J = df(x, p...);
        return J;
    }

    template<class TypeFuncstateTransition,
             class ...TypeParam>
    M jacobianMatrices(const M& x,
                 TypeFuncstateTransition f,
                 std::nullptr_t df,
                 TypeParam ...p)
    {
        M J = numericJacobianAdditive(f, x, p...);
        return J;
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
        M J = jacobianMatrices(x, h, dh, p...);
        ans.Rsqrt = Rs;
        ans.dHdx = J;
        ans.zEstimated = h(x, p...);
        ans.Pxy = (S*Utils::transpose(S))*Utils::transpose(ans.dHdx);
        ans.Sy = Utils::qrFactor_A(ans.dHdx, S, ans.Rsqrt);
        return ans;
    }

    std::pair<M,M> correctstateAndSqrtCovariance(const M& x,
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

template<class M, class SM, class MM, class GM>
struct EKF_QR : public EKF_QRMath<M>, public Filter<M>
{
private:
    M state;
    M covariance;
    M process_noise;
    M measurement_noise;

    M state_predict;
    M covariance_predict;
    M measurement_predict;
    M covariance_of_measurement_predict;
    M residue;

public:
    EKF_QR(M state,M covariance,M process_noise,M measureNoise):
        state(state),
        process_noise(process_noise),
        covariance(covariance),
        measurement_noise(measureNoise)
    {}

    EKF_QR(const EKF_QR& ekf_qr):
        state(ekf_qr.state),
        covariance(ekf_qr.covariance),
        process_noise(ekf_qr.process_noise),
        measurement_noise(ekf_qr.measurement_noise),
        state_predict(ekf_qr.state_predict),
        covariance_predict(ekf_qr.covariance_predict),
        measurement_predict(ekf_qr.measurement_predict),
        covariance_of_measurement_predict(ekf_qr.covariance_of_measurement_predict),
        residue(ekf_qr.residue)
    {}

    M getState()const override{return state;}
    M getCovariance()const override{return covariance;}
    M getStatePredict()const override{return state_predict;}
    M getCovariancePredict()const override{return covariance_predict;}
    M getMeasurementPredict()const override{return measurement_predict;}
    M getCovarianceOfMeasurementPredict()const override{return covariance_of_measurement_predict;}
    bool setState(M& state)override{state = state;return true;}
    bool setCovariance(M& covariance)override{covariance = covariance;return true;}

    template <class ...TP>
    std::pair<M,M> predict(double dt,TP ...p)
    {
        GM Gm;
        M GG = Gm(dt);
        M Q = GG*process_noise*Utils::transpose(GG);
        M sqrtprocess_noise = Utils::sqrtMatrix(Q);
        M sqrtcovariance = Utils::sqrtMatrix(covariance);

        auto pred = EKF_QRMath<M>::predict(sqrtprocess_noise,
                                        state,
                                        sqrtcovariance,
                                        SM(),
                                        p...,
                                        dt);
        state = pred.x;
        covariance = Utils::FromSqrtMatrix(pred.S);

        state_predict = pred.x;
        covariance_predict = Utils::FromSqrtMatrix(pred.S);

        return std::make_pair(state,covariance);
    }

    template <class ...TypeParam>
    std::pair<M,M> correct(const M& measurement,TypeParam ...param)
    {
        M sqrtcovariance = Utils::sqrtMatrix(this->covariance);
        M sqrtmeasurement_noise = Utils::sqrtMatrix(this->measurement_noise);

        auto corr = EKF_QRMath<M>::correct(measurement,
                                        measurement_predict,
                                        covariance_of_measurement_predict,
                                        residue,
                                        sqrtmeasurement_noise,
                                        state,
                                        sqrtcovariance,
                                        MM(),
                                        nullptr,
                                        measurement//#TODO - сделать через параметр
                                        /*param...*/);
        state = corr.first;
        covariance = Utils::FromSqrtMatrix(corr.second);

        return std::make_pair(state,covariance);
    }
};
}





