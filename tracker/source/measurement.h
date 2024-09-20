#pragma once

#include "utils.h"

namespace Tracker
{

struct Measurement
{
    int id;
    double timepoint;
    double x;
    double y;
    double z;
    double vx;
    double vy;
    double vz;
    double ax;
    double ay;
    double az;
    double w;
    double std_meas_x;
    double std_meas_y;
    double std_meas_z;
    double std_meas_velo_x;
    double std_meas_velo_y;
    double std_meas_velo_z;
};

template<class M>
struct Measurement3 : public Measurement
{
    int id() const {return Measurement::id;}
    double timepoint() const {return Measurement::timepoint;}
    double x(){return Measurement::x;}
    double y(){return Measurement::y;}
    double z(){return Measurement::z;}
    double vx() const {return 0.;}
    double vy() const {return 0.;}
    double vz() const {return 0.;}
    double ax() const {return 0.;}
    double ay() const {return 0.;}
    double az() const {return 0.;}
    double w() const {return 0.;}
    double std_meas_x() const {return Measurement::std_meas_x;}
    double std_meas_y() const {return Measurement::std_meas_y;}
    double std_meas_z() const {return Measurement::std_meas_z;}
    double std_meas_velo_x() const {return 0.;}
    double std_meas_velo_y() const {return 0.;}
    double std_meas_velo_z() const {return 0.;}
    M get() const
    {
        M m(3,1);
        m << Measurement::x,Measurement::y,Measurement::z;
        return m;
    }
};

template<class M>
struct Measurement3R : public Measurement //#TEMP SOLUTION
{
    int id() const {return Measurement::id;}
    double timepoint() const {return Measurement::timepoint;}
    double x(){return Measurement::x;}
    double y(){return Measurement::y;}
    double z(){return Measurement::z;}
    double vx() const {return 0.;}
    double vy() const {return 0.;}
    double vz() const {return 0.;}
    double ax() const {return 0.;}
    double ay() const {return 0.;}
    double az() const {return 0.;}
    double w() const {return 0.;}
    double std_meas_x() const {return Measurement::std_meas_x;}
    double std_meas_y() const {return Measurement::std_meas_y;}
    double std_meas_z() const {return Measurement::std_meas_z;}
    double std_meas_velo_x() const {return 0.;}
    double std_meas_velo_y() const {return 0.;}
    double std_meas_velo_z() const {return 0.;}
    M get() const
    {
        M m(3,1);
        m << Measurement::x,Measurement::y,Measurement::z;
        return m;
    }
};

template<class M>
struct Measurement2 : public Measurement
{
    int id() const {return Measurement::id;}
    double timepoint() const {return Measurement::timepoint;}
    double x() const {return Measurement::x;}
    double y() const {return Measurement::y;}
    double z() const {return 0.;}
    double vx() const {return 0.;}
    double vy() const {return 0.;}
    double vz() const {return 0.;}
    double ax() const {return 0.;}
    double ay() const {return 0.;}
    double az() const {return 0.;}
    double w() const {return 0.;}
    double std_meas_x() const {return Measurement::std_meas_x;}
    double std_meas_y() const {return Measurement::std_meas_y;}
    double std_meas_z() const {return 0.;}
    double std_meas_velo_x() const {return 0.;}
    double std_meas_velo_y() const {return 0.;}
    double std_meas_velo_z() const {return 0.;}
    M get() const
    {
        M m(2,1);
        m << Measurement::x,Measurement::y;
        return m;
    }
};

}
