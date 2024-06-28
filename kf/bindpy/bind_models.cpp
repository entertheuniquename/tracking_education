#include "bind_models.h"

void bind_models(pybind11::module &m)
{
    m.def("BindFCV_10", [](Eigen::MatrixXd x,double dt){
        Models10::FCV<Eigen::MatrixXd> fcv;
        return fcv(x,dt);});
    m.def("BindFCA_10", [](Eigen::MatrixXd x,double dt){
        Models10::FCA<Eigen::MatrixXd> fca;
        return fca(x,dt);});
    m.def("BindFCT_10", [](Eigen::MatrixXd x,double dt){
        Models10::FCT<Eigen::MatrixXd> fct;
        return fct(x,dt);});
    m.def("BindHXX_10", [](){
        Models10::H<Eigen::MatrixXd> h;
        return h();});
    m.def("BindHXX_10", [](Eigen::MatrixXd x){
        Models10::H<Eigen::MatrixXd> h;
        return h(x);});
    m.def("BindG_10", [](double t){
        Models10::G<Eigen::MatrixXd> g;
        return g(t);});
    m.def("BindFCVJ_10", [](Eigen::MatrixXd x, double t){
          Models10::FCV_Jacobian<Eigen::MatrixXd> mm;
          return mm(x,t); });
    m.def("BindFCTJ_10", [](Eigen::MatrixXd x, double t){
          Models10::FCT_Jacobian<Eigen::MatrixXd> mm;
          return mm(x,t); });
    m.def("BindHJ_10", [](Eigen::MatrixXd x){//#TODO - заменить на BindHXXJ
          Models10::H_Jacobian<Eigen::MatrixXd> mm;
          return mm(x); });
    //#LEGACY
    m.def("stateModel_CV", [](double dt){
        Models::StateModel_CV<Eigen::MatrixXd> sm;
        return sm(dt);});
    m.def("stateModel_CVx", [](Eigen::MatrixXd x,double dt){
        Models::StateModel_CV<Eigen::MatrixXd> sm;
        return sm(x,dt);});
    m.def("stateModel_CTx", [](Eigen::MatrixXd x,double dt){
        Models::StateModel_CT<Eigen::MatrixXd> sm;
        return sm(x,dt);});
    m.def("stateModel_CT6", [](Eigen::MatrixXd x){
        std::cout << "stateModel_CT6" << std::endl;
        Models::StateModel_CT<Eigen::MatrixXd> sm;
        return sm(x,6.);});
    m.def("measureModel_XX", [](){
          Models::MeasureModel_XvXYvYZvZ_XYZ<Eigen::MatrixXd> mm;
          return mm();});
    m.def("measureModel_XXx", [](Eigen::MatrixXd x){
          Models::MeasureModel_XvXYvYZvZ_XYZ<Eigen::MatrixXd> mm;
          return mm(x);});
    m.def("measureModel_XwX", [](){
          Models::MeasureModel_XvXYvYZvZW_XYZ<Eigen::MatrixXd> mm;
          return mm();});
    m.def("measureModel_XwXx", [](Eigen::MatrixXd x){
          Models::MeasureModel_XvXYvYZvZW_XYZ<Eigen::MatrixXd> mm;
          return mm(x);});
    m.def("measureModel_XRx", [](Eigen::MatrixXd x){
          Models::MeasureModel_XvXYvYZvZ_EAR<Eigen::MatrixXd> mm;
          return mm(x); });
    m.def("FJacobian_CT", [](Eigen::MatrixXd x, double t){
          Models::FJacobian_CT<Eigen::MatrixXd> mm;
          return mm(x,t); });
    m.def("FCTJ10", [](Eigen::MatrixXd x, double t){
          Models10::FCT_Jacobian<Eigen::MatrixXd> mm;
          return mm(x,t); });
    m.def("FCT10", [](Eigen::MatrixXd x, double t){
          Models10::FCT<Eigen::MatrixXd> mm;
          return mm(x,t); });
    m.def("FCTdegJ10", [](Eigen::MatrixXd x, double t){
          Models10::FCT_deg_Jacobian<Eigen::MatrixXd> mm;
          return mm(x,t); });
    m.def("FCTdeg10", [](Eigen::MatrixXd x, double t){
          std::cout << "FCTdeg10:" << std::endl;
          //std::cout << "-> " << std::endl << x << std::endl << t << std::endl;
          Models10::FCT_deg<Eigen::MatrixXd> mm;
          return mm(x,t); });
    m.def("H10", [](Eigen::MatrixXd x){
          Models10::H<Eigen::MatrixXd> mm;
          return mm(x); });
    m.def("HJ10", [](Eigen::MatrixXd x){
          Models10::H_Jacobian<Eigen::MatrixXd> mm;
          return mm(x); });
}
