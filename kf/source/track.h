#pragma once

#include "estimator.h"

struct Measurement
{
    Point p;
    double t; //timepoint
};

template<class M>
class Track : public Estimator
{
private:
    Point point;
public:
    Track(Measurement measurement,
          M in_covariance,
          M in_transition_state_model,
          M in_process_noise,
          M in_transition_process_noise_model,
          M in_transition_measurement_model,
          M in_measurement_noise):
          Estimator(M{{measurement.p.x,measurement.p.vx,measurement.p.y,measurement.p.vy,measurement.p.z,measurement.p.vz}},
                  in_covariance,
                  in_transition_state_model,
                  in_process_noise,
                  in_transition_process_noise_model,
                  in_transition_measurement_model,
                  in_measurement_noise)
    {

    }

    void step(const Measurement& measurement)
    {
        pred();
        corr(measurement.p,measurement.t);
    }

    void step(double timepoint)
    {
        pred();
    }
};
