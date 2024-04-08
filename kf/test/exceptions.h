#pragma once

#include <Eigen/Dense>
namespace Exceptions
{
inline void detx(Eigen::MatrixXd X)
{
    if(X.determinant()==0)
        throw 66;
}
inline void bad_prob_value(double x)
{
    if(x<0 || x>1)
        throw 55;
}
}
