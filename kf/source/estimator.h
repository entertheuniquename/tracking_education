#pragma once

#include <utility>

template<class M>
class Estimator
{
public:
    virtual std::pair<M,M> predict(const M& in_transition_state_model,
                                   const M& in_transition_measurement_model,
                                   const M& in_control_input,
                                   const M& in_control_model)=0;
    virtual std::pair<M,M> correct(const M& in_transition_measurement_model,
                                   const M& in_measurement)=0;
};
