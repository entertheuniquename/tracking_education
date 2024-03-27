#pragma once

#include "utils.h"

namespace MathematicsCommon
{
template<class M>
void kf_predict(M& xp,const M& A,const M& x,const M& B,const M& u,
                M& Pp,const M& P,const M& G,const M& Q,
                M& zp,const M& H)
{
    xp = A*x + B*u;
    Pp = A*P*Utils::transpose(A) + G*Q*Utils::transpose(G);
    Pp = (Pp + Utils::transpose(Pp))/2.;
    zp = H*xp;
}

template<class M>
void kf_correct(const M& H,const M& Pp,const M& R,
                M& xc,const M& xp,const M& z,const M& zp,
                M& Pc)
{
    M S = H*Pp*Utils::transpose(H) + R;
    M K = Pp*Utils::transpose(H)*Utils::inverse(S);
      xc = xp + K * (z - zp);
      Pc = Pp - K * S * Utils::transpose(K);
      Pc = (Pc + Utils::transpose(Pc))/2.;
}
}

