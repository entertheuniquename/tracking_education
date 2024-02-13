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

// for [x,vx,y,vy,z,vz]
template <class M>
M stateModel3A(const M& x, double T) {//[D2]
    //std::cout << "x:" << x << std::endl;
    M F = {{1, T, 0, 0, 0, 0},
           {0, 1, 0, 0, 0, 0},
           {0, 0, 1, T, 0, 0},
           {0, 0, 0, 1, 0, 0},
           {0, 0, 0, 0, 1, T},
           {0, 0, 0, 0, 0, 1}};
    return F*x;
};//remake 2

//// for [x,y,z,vx,vy,vz]
//template <class M>
//M stateModel3B(const M& x, double T) {
//    //std::cout << "x:" << x << std::endl;
//    M F = {{1, 0, 0, 0, 0, 0},
//           {0, 1, 0, 0, 0, 0},
//           {0, 0, 1, 0, 0, 0},
//           {T, 0, 0, 1, 0, 0},
//           {0, T, 0, 0, 1, 0},
//           {0, 0, T, 0, 0, 1}};
//    return F*x;
//};//3B

// for [x,y,z,vx,vy,vz]
template <class M>
M stateModel3B(const M& x, double T) {
    //std::cout << "x:" << x << std::endl;
    M F = {{1, 0, 0, T, 0, 0},
           {0, 1, 0, 0, T, 0},
           {0, 0, 1, 0, 0, T},
           {0, 0, 0, 1, 0, 0},
           {0, 0, 0, 0, 1, 0},
           {0, 0, 0, 0, 0, 1}};
    return F*x;
};//3B

template <class M>
M measureModel(const M& x, const M& z = M{}) {
    double angle = atan2(x(2), x(0));
    double range = sqrt(x(0)*x(0) + x(2)*x(2));
    if (!z.empty()) {
        angle = z(0) + Utils::ComputeAngleDifference(angle, z(0));
    }
    M r = {angle, range};
    return trans(r);
};

// for [x,vx,y,vy,z,vz]
template <class M>
M measureModel3A(const M& x, const M& z = M{}) {
    double elev = atan2(x(4), sqrt(x(2)*x(2) + x(0)*x(0)));
    double angle = atan2(x(2), x(0));
    double range = sqrt(x(0)*x(0) + x(2)*x(2) + x(4)*x(4));
    if (!z.empty()) {
        angle = z(1) + Utils::ComputeAngleDifference(angle, z(1));
    }
    M r = {elev, angle, range};
    //std::cout << "x: " << x(0) << "/" << x(1) << " y: " << x(2) << "/" << x(3) << " z: " << x(4) << "/" << x(5) << std::endl;
    //std::cout << "e: " << elev << " a: " << angle << " r: " << range << std::endl;
    //std::cout << "r:" << r << std::endl;
    //std::cout << "r:" << trans(r) << std::endl;
    return trans(r);
};//remake 2

// for [x,y,z,vx,vy,vz]
template <class M>
M measureModel3B(const M& x, const M& z = M{}) {
    double elev = atan2(x(2), sqrt(x(1)*x(1) + x(0)*x(0)));
    double angle = atan2(x(1), x(0));
    double range = sqrt(x(0)*x(0) + x(1)*x(1) + x(2)*x(2));
    if (!z.empty()) {
        angle = z(1) + Utils::ComputeAngleDifference(angle, z(1));
    }
    M r = {elev, angle, range};
    //std::cout << "x: " << x(0) << "/" << x(1) << " y: " << x(2) << "/" << x(3) << " z: " << x(4) << "/" << x(5) << std::endl;
    //std::cout << "e: " << elev << " a: " << angle << " r: " << range << std::endl;
    //std::cout << "r:" << r << std::endl;
    //std::cout << "r:" << trans(r) << std::endl;
    return trans(r);
};

}
