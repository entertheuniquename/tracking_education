#pragma once

#include "Eigen/Dense"

namespace MathematicsCommon
{
Eigen::MatrixXd transpose(Eigen::MatrixXd in)
{
    return in.transpose();
}

Eigen::MatrixXd inverse(Eigen::MatrixXd in)
{
    return in.inverse();
}

template<class M>
void kf_predict(M& xp,const M& A,const M& x,const M& B,const M& u,
                M& Pp,const M& P,const M& G,const M& Q,
                M& zp,const M& H)
{
    xp = A*x + B*u;
    Pp = A*P*transpose(A) + G*Q*transpose(G);
    Pp = (Pp + transpose(Pp))/2.;
    zp = H*xp;
}

template<class M>
void kf_correct(const M& H,const M& Pp,const M& R,
                M& xc,const M& xp,const M& z,const M& zp,
                M& Pc)
{
    M S = H*Pp*transpose(H) + R;
    M K = Pp*transpose(H)*inverse(S);
      xc = xp + K * (z - zp);
      Pc = Pp - K * S * transpose(K);
      Pc = (Pc + transpose(Pc))/2.;
}
}

