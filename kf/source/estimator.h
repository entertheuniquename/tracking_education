#pragma once

#include <utility>

struct Point
{
    double x;
    double vx;
    double y;
    double vy;
    double z;
    double vz;
};

class Estimator
{
public:
    virtual Point pred()=0;
    virtual Point corr(Point p, double t)=0;
};
