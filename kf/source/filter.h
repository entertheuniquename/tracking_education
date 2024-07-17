#pragma once

#include <utility>

namespace Estimator
{
template<class M>
class Filter
{
public:
    virtual M getState() const=0;
    virtual M getCovariance() const=0;
    virtual M getStatePredict() const=0;
    virtual M getCovariancePredict() const=0;
    virtual M getMeasurementPredict() const=0;
    virtual M getCovarianceOfMeasurementPredict() const=0;
    bool virtual setState(M& in)=0;
    bool virtual setCovariance(M& in)=0;

    virtual std::pair<M,M> predict(double dt)=0;
    virtual std::pair<M,M> correct(const M& in_measurement)=0;
};

}
