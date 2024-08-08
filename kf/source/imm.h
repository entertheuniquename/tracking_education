#pragma once

#include "filter.h"
#include "models.h"
#include <memory>

namespace Estimator
{
template <class M>
struct IMMMath
{
public:
    M mixing_probability(int length, M mu, M tp)
    {
        M mx(length,length);
        mx.setZero();
        M fi(1,length);
        fi.setZero();

        for(int j=0;j<length;j++)
            for(int i=0;i<length;i++)
                fi(j)+=mu(i)*tp(i,j);


        for(int j=0;j<length;j++)
            for(int i=0;i<length;i++)
                mx(i,j)=tp(i,j)*mu(i)/fi(j);

        return mx;
    }

    long double system_probability(M z,M ze,M Se,int length)
    {
        M residual = z-ze;
        M power = -0.5*residual*Utils::inverse(Se)*Utils::transpose(residual);
        return (1./std::sqrt(std::pow(2.*M_PI,length)*Se.determinant()))*std::exp(power.determinant());
    }
};

template<class M, class ...Estimators>
class IMM : public IMMMath<M>, public Filter<M>
{
private:
    M state;
    M covariance;
    M state_predict;
    M covariance_predict;
    //std::vector<std::unique_ptr<Filter<M>>> estimators; //#TODO - smart ptr
    std::vector<Filter<M>*> estimators;
    M mu;
    M tp;
public:
    static constexpr std::size_t length = sizeof...(Estimators);
    int lenx0;

    IMM(M m, M t, Estimators ...eee):
        mu(m),
        tp(t),
        //estimators{std::make_unique<Filter<M>>(eee)...}, //#TODO - smart ptr
        estimators{new Estimators(eee)...},
        lenx0(estimators.at(0)->getState().rows())
    {}

    IMM(M m, M t, M x0, M P0, M Q0, M R):
        mu(m),
        tp(t),
        //estimators{std::make_unique<Filter<M>>(Estimators(x0,P0,Q0,R))...}, //#TODO - smart ptr
        estimators{new Estimators(x0,P0,Q0,R)...},
        lenx0(estimators.at(0)->getState().rows())
    {}

    IMM(const IMM& imm):
        state(imm.state),
        covariance(imm.covariance),
        state_predict(imm.state_predict),
        covariance_predict(imm.covariance_predict),
        //measurement_predict(imm.measurement_predict), //#TODO - сделать правильную инициализацию по образу остальных фильтров
        //covariance_of_measurement_predict(imm.covariance_of_measurement_predict), //#TODO - сделать правильную инициализацию по образу остальных фильтров
        //residue(imm.residue) //#TODO - сделать правильную инициализацию по образу остальных фильтров
        mu(imm.mu),
        tp(imm.tp),
        estimators{new Estimators(&imm.estimators)...}
    {}

    ~IMM(){for(auto e : estimators){/*std::cout << "imm.filter[deleted]" << std::endl;*/delete e;}estimators.clear();}

    std::vector<std::pair<M,M>> mixed_states_and_covariances()
    {
        M mx = IMMMath<M>::mixing_probability(length,mu,tp);

        std::vector<std::pair<M,M>> xp;

        for(auto filter : estimators)
            xp.push_back(std::make_pair(filter->getState(),filter->getCovariance()));

        std::vector<std::pair<M,M>> x0p0;

        for(int j=0;j<length;j++)
        {
            M xj(lenx0,1);
            xj.setZero();
            M p0(lenx0,lenx0);
            p0.setZero();
            for(int i=0;i<length;i++)
               xj+=mx(i,j)*xp[i].first;
            x0p0.push_back(std::make_pair(xj,p0));
        }

        for(int j=0;j<length;j++)
        {
            M Pj(lenx0,lenx0);
            Pj.setZero();
            for(int i=0;i<length;i++)
                Pj += mx(i,j)*(xp.at(i).second+(xp.at(i).first-x0p0.at(j).first)*Utils::transpose(xp.at(i).first-x0p0.at(j).first));
            x0p0[j].second = Pj;
        }

        return x0p0;
    }

    M models_probability_recalculation(M& z)
    {
        M mx = IMMMath<M>::mixing_probability(length,mu,tp);

        std::vector<std::pair<M,M>>zs;

        for(auto filter : estimators)
            zs.push_back(std::make_pair(filter->getMeasurementPredict(),filter->getCovarianceOfMeasurementPredict()));

        std::vector<long double>mu0;

        for(int j=0;j<length;j++)
        {
            long double mu00=0;
            for(int i=0;i<length;i++)
                mu00+=mx(i,j)*mu(i);
            M zp = Utils::transpose(zs.at(j).first);
            M Se = zs.at(j).second;
            long double sp = IMMMath<M>::system_probability(z,zp,Se,length);
            mu00*=sp;
            mu0.push_back(mu00);
        }

        long double denominator = 0.;
        for(int j=0;j<length;j++)
            denominator+=mu0.at(j);

        std::vector<long double>mux;

        for(int j=0;j<length;j++)
            mux.push_back(mu0.at(j)/denominator);

        for(int j=0;j<length;j++)
            mu(j) = mux.at(j);

        return mu;
    }

    std::pair<M,M> combine_state_and_covariance()//#TODO - проверить на безопасность!
    {
        std::vector<std::pair<M,M>> xp;

        for(auto filter : estimators)
            xp.push_back(std::make_pair(filter->getState(),filter->getCovariance()));

        M cs(lenx0,1);
        cs.setZero();

        for(int j=0;j<length;j++)
            cs+=mu(j)*xp.at(j).first;

        M cc(lenx0,lenx0);
        cc.setZero();

        for(int j=0;j<length;j++)
            cc+=mu(j)*(xp.at(j).second+(xp.at(j).first-cs)*Utils::transpose(xp.at(j).first-cs));

        return std::make_pair(cs,cc);
    }

    std::pair<M,M> predict(double dt)
    {
        //calculate mixed states and covariances
        std::vector<std::pair<M,M>> mxs = mixed_states_and_covariances();
        //set mixed states and covariaces
        for(int i=0;i<estimators.size();i++)
        {
            estimators.at(i)->setState(mxs.at(i).first);
            estimators.at(i)->setCovariance(mxs.at(i).second);
            estimators.at(i)->predict(dt);
        }
        //predicts
        auto comb = combine_state_and_covariance();
        state = state_predict = comb.first;
        covariance = covariance_predict = comb.second;

        return std::make_pair(state,covariance);
    }

    std::pair<M,M> correct(const M& measurement)
    {
        //corrects
        for(int i=0;i<estimators.size();i++)
            estimators.at(i)->correct(measurement);

        //recalculate probabilities
        M z = Utils::transpose(measurement);
        models_probability_recalculation(z);

        //calculate combine state and covariance
        auto comb = combine_state_and_covariance();
        state = comb.first;
        covariance = comb.second;

        return std::make_pair(state,covariance);
    }

public:
    M getState()const override{return state;}
    M getCovariance()const override{return covariance;}
    M getStatePredict()const override{return state_predict;}
    M getCovariancePredict()const override{return covariance_predict;}
    M getMeasurementPredict()const override{return M();}//#ZAGL //#TODO - сделать правильный возврат
    std::pair<M,M> getMeasurementPredictData(double dt)const override{return std::make_pair(M(),M());/*#ZAGL*/}
    M getCovarianceOfMeasurementPredict()const override{return M();}//#ZAGL //#TODO - сделать правильный возврат
    bool setState(M& state_in)override{state = state_in;return true;}
    bool setCovariance(M& covariance_in)override{covariance = covariance_in;return true;}

    M getMU()const{return mu;}

};
}
