#pragma once

namespace Models
{
enum class X3A{X=0,VX=1,Y=2,VY=3,Z=4,VZ=5};
enum class X3B{VZ=0,Y=1,Z=2,VY=3,VX=4,X=5};
enum class Z{X=0,Y=1,Z=2};
template <class M>
M stateModel_3A(double T)
{
    M F(6,6);
    F << 1., T , 0., 0., 0., 0.,
         0., 1., 0., 0., 0., 0.,
         0., 0., 1., T , 0., 0.,
         0., 0., 0., 1., 0., 0.,
         0., 0., 0., 0., 1., T ,
         0., 0., 0., 0., 0., 1.;
    return F;
};

template <class M>
M stateModel_3B(double T)
{
    M F(6,6);
    F << 1., 0., 0., 0., 0., 0.,
         0., 1., 0., T , 0., 0.,
         T , 0., 1., 0., 0., 0.,
         0., 0., 0., 1., 0., 0.,
         0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., T , 1.;
    return F;
};

template <class M>
M measureModel_3A() {
    M H(3,6);
    H << 1., 0., 0., 0., 0., 0.,
         0., 0., 1., 0., 0., 0.,
         0., 0., 0., 0., 1., 0.;
    return H;
};

template <class M>
M measureModel_3B() {
    M H(3,6);
    H << 0., 0., 0., 0., 0., 1.,
         0., 1., 0., 0., 0., 0.,
         0., 0., 1., 0., 0., 0.;
    return H;
};

template <class M>
M GModel_3A(double T) {
    M G(6,3);
    G <<   T*T/2.,       0.,       0.,
               T ,       0.,       0.,
               0.,   T*T/2.,       0.,
               0.,       T ,       0.,
               0.,       0.,   T*T/2.,
               0.,       0.,       T ;
    return G;
};

template <class M>
M GModel_3B(double T) {
    M G(6,3);
    G <<       0.,       0.,       T ,
               0.,   T*T/2.,       0.,
               0.,       0.,   T*T/2.,
               0.,       T ,       0.,
               T ,       0.,       0.,
           T*T/2.,       0.,       0.;
    return G;
};

template <class M>
M HposModel_3A() {
    M Hpos(3,6);
    Hpos << 1., 0., 0., 0., 0., 0.,
            0., 0., 1., 0., 0., 0.,
            0., 0., 0., 0., 1., 0.;
    return Hpos;
};
template <class M>
M HvelModel_3A() {
    M Hpos(3,6);
    Hpos << 0., 1., 0., 0., 0., 0.,
            0., 0., 0., 1., 0., 0.,
            0., 0., 0., 0., 0., 1.;
    return Hpos;
};

template <class M>
M HposModel_3B() {
    M Hpos(3,6);
    Hpos << 0., 0., 0., 0., 0., 1.,
            0., 1., 0., 0., 0., 0.,
            0., 0., 1., 0., 0., 0.;
    return Hpos;
};
template <class M>
M HvelModel_3B() {
    M Hvel(3,6);
    Hvel << 0., 0., 0., 0., 1., 0.,
            0., 0., 0., 1., 0., 0.,
            1., 0., 0., 0., 0., 0.;
    return Hvel;
};
}
