#pragma once

#include "utils.h"

#include <armadillo>

namespace Models {

template <class M>
M stateModel(const M& x, double T) {
    M F = {{1, T, 0, 0},
           {0, 1, 0, 0},
           {0, 0, 1, T},
           {0, 0, 0, 1}};
    return F*x;
};

template <class M>
M stateModel3A(const M& x, double T) {
    M F = {{1, T, 0, 0, 0, 0},
           {0, 1, 0, 0, 0, 0},
           {0, 0, 1, T, 0, 0},
           {0, 0, 0, 1, 0, 0},
           {0, 0, 0, 0, 1, T},
           {0, 0, 0, 0, 0, 1}};
    return F*x;
};

template <class M>
M stateModel3B(const M& x, double T) {
    M F = {{1, 0, 0, T, 0, 0},
           {0, 1, 0, 0, T, 0},
           {0, 0, 1, 0, 0, T},
           {0, 0, 0, 1, 0, 0},
           {0, 0, 0, 0, 1, 0},
           {0, 0, 0, 0, 0, 1}};
    return F*x;
};

template <class M>
M measureModel(const M& x, const M& z = M{}) {
    enum class POSITION{X=0,VX=1,Y=2,VY=3};
    double X = x(static_cast<double>(POSITION::X));
    double Y = x(static_cast<double>(POSITION::Y));
    double angle = atan2(Y,X);
    double range = sqrt(X*X+Y*Y);
    if (!z.empty()) {
        angle = z(0) + Utils::ComputeAngleDifference(angle, z(0));
    }
    M r = {angle, range};
    return trans(r);
};

template <class M>
M measureModel3A(const M& x, const M& z = M{}) {
    enum class POSITION{X=0,VX=1,Y=2,VY=3,Z=4,VZ=5};
    double X = x(static_cast<double>(POSITION::X));
    double Y = x(static_cast<double>(POSITION::Y));
    double Z = x(static_cast<double>(POSITION::Z));
    double elev = atan2(Z, sqrt(Y*Y+X*X));
    double angle = atan2(Y,X);
    double range = sqrt(X*X+Y*Y+Z*Z);
    if (!z.empty()) {
        angle = z(1) + Utils::ComputeAngleDifference(angle, z(1));
    }
    M r = {elev, angle, range};
    return trans(r);
};

template <class M>
M measureModel3B(const M& x, const M& z = M{}) {
    enum class POSITION{X=0,Y=1,Z=2,VX=3,VY=4,VZ=5};
    double X = x(static_cast<double>(POSITION::X));
    double Y = x(static_cast<double>(POSITION::Y));
    double Z = x(static_cast<double>(POSITION::Z));
    double elev = atan2(Z, sqrt(Y*Y+X*X));
    double angle = atan2(Y,X);
    double range = sqrt(X*X+Y*Y+Z*Z);
    if (!z.empty()) {
        angle = z(1) + Utils::ComputeAngleDifference(angle, z(1));
    }
    M r = {elev, angle, range};
    return trans(r);
};

}
