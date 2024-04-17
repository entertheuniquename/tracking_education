#pragma once

#include "kf_eigen3.h"
#include <map>
#include <string>
#include <memory>
#include "models.h"

//#[x,vx,y,vy,z,vz] //#TODO - X3A - потенциал для расширения
template<class M>
struct Measurement
{
    double timepoint;
    M point;
    M measurement_noise;
    M process_noise;
};

//#[x,vx,y,vy,z,vz] //#TODO - X3A - потенциал для расширения
//#kf //#TODO - KFE - потенциал для расширения
template<class M, class EstimatorType>
class Track
{
private:
    std::unique_ptr<EstimatorType> estimator; //#TODO - KFE - потенциал для расширения
    Measurement<M> measurement;
public:
    Track(Measurement<M> in_measurement,
          M in_covariance,
          M in_transition_state_model,
          M in_process_noise,
          M in_transition_process_noise_model,
          M in_transition_measurement_model):
        measurement(in_measurement)
    {
        estimator = std::make_unique<EstimatorType>(in_transition_measurement_model.transpose()*in_measurement.point,//#BAD_VARIANT
                                                    in_covariance,
                                                    in_transition_state_model,
                                                    in_process_noise,
                                                    in_transition_process_noise_model,
                                                    in_transition_measurement_model,
                                                    in_measurement.measurement_noise);
    }
    Measurement<M> step(const Measurement<M>& m) //#TODO - аргументы - потенциал для расширения
    {

        double dt = m.timepoint - measurement.timepoint;

        M new_transition_state_model = Models::stateModel_3A<M>(dt); //#TODO - X3A - потенциал для расширения
        M new_transition_process_noise_model = Models::GModel_3A<M>(dt); //#TODO - X3A - потенциал для расширения
        M new_transition_measurement_model = Models::measureModel_3A<M>(); //#TODO - X3A - потенциал для расширения
        M new_measurement = m.point;
        M new_measurement_noise = m.measurement_noise;

        estimator->predict(new_transition_state_model,
                           new_transition_process_noise_model,
                           new_transition_measurement_model);

        estimator->correct(new_transition_measurement_model,
                           new_measurement,
                           new_measurement_noise);

        measurement.timepoint = m.timepoint;
        measurement.point = estimator->get_state();
        measurement.measurement_noise = m.measurement_noise;

        return measurement;
    }
    Measurement<M> step(double timepoint) //#TODO - аргументы - потенциал для расширения
    {
        double dt = timepoint - measurement.timepoint;

        M new_transition_state_model = Models::stateModel_3A<M>(dt); //#TODO - X3A - потенциал для расширения
        M new_transition_process_noise_model = Models::GModel_3A<M>(dt); //#TODO - X3A - потенциал для расширения
        M new_transition_measurement_model = Models::measureModel_3A<M>(); //#TODO - X3A - потенциал для расширения

        estimator->predict(new_transition_state_model,
                           new_transition_process_noise_model,
                           new_transition_measurement_model);

        measurement.timepoint = timepoint;
        measurement.point = estimator->get_state();

        return measurement;
    }
    double get_timepoint(){return measurement.timepoint;}
};
