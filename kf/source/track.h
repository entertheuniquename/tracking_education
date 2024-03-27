#pragma once

#include "estimator.h"

struct Point
{
    double x;
    double vx;
    double y;
    double vy;
    double z;
    double vz;
};

struct Measurement
{
    Point p;
    double t; //timepoint
};

template<class M>
class Track : public Estimator<M>
{
private:
    Point point;
public:
    Track()/*:Estimator()*/ //#TODO
    {

    }

    void step(const Measurement& measurement)
    {
        //predict();//#TODO
        //correct();//#TODO
    }

    void step(double timepoint)
    {
        //predict();//#TODO
    }
};
