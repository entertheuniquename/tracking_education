#pragma once

namespace MathematicsCommon
{
template<class M>
void kf_predict(M& xp,const M& A,const M& x,const M& B,const M& u,
                M& Pp,const M& P,const M& G,const M& Q,
                M& zp,const M& H)
{
    xp = A*x + B*u;
    Pp = A*P*A.transpose() + G*Q*G.transpose();
    Pp = (Pp + Pp.transpose())/2.;
    zp = H*xp;
}

template<class M>
void kf_correct(const M& H,const M& Pp,const M& R,
                M& xc,const M& xp,const M& z,const M& zp,
                M& Pc)
{
    M S = H*Pp*H.transpose() + R;
    M K = Pp*H.transpose()*S.inverse();
      xc = xp + K * (z - zp);
      Pc = Pp - K * S * K.transpose();
      Pc = (Pc + Pc.transpose())/2.;
}
}
