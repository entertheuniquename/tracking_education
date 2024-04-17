#include "bind_kf.h"

namespace py = pybind11;

class BindKFE
{
private:
    KFE kfe;
public:

    BindKFE(Eigen::MatrixXd in_state,
            Eigen::MatrixXd in_covariance,
            Eigen::MatrixXd in_transition_state_model,
            Eigen::MatrixXd in_process_noise,
            Eigen::MatrixXd in_transition_process_noise_model,
            Eigen::MatrixXd in_transition_measurement_model,
            Eigen::MatrixXd in_measurement_noise):
        kfe(in_state,
            in_covariance,
            in_transition_state_model,
            in_process_noise,
            in_transition_process_noise_model,
            in_transition_measurement_model,
            in_measurement_noise){}

//#TODO
//    /*py::tuple*/Eigen::MatrixXd predKFE(const Eigen::MatrixXd &a,const Eigen::MatrixXd &q,const Eigen::MatrixXd &g,const Eigen::MatrixXd &m,const Eigen::MatrixXd &u,const Eigen::MatrixXd &b)
//    {
//        std::cout << std::endl << "predKFE: a" << std::endl
//                  << a <<  std::endl;
//        std::cout << std::endl << "predKFE: q" << std::endl
//                  << q <<  std::endl;
//        std::cout << std::endl << "predKFE: g" << std::endl
//                  << g <<  std::endl;
//        std::cout << std::endl << "predKFE: m" << std::endl
//                  << m <<  std::endl;
//        std::cout << std::endl << "predKFE: u" << std::endl
//                  << u <<  std::endl;
//        std::cout << std::endl << "predKFE: b" << std::endl
//                  << b <<  std::endl;
//        auto ret = kfe.predict(a,q,g,m,u,b);
//        std::cout << std::endl << "predKFE" << std::endl;
//        return /*py::make_tuple(*/ret.first/*,ret.second)*/;
//    }
    /*py::tuple*/Eigen::MatrixXd predKFE(const Eigen::MatrixXd &a,const Eigen::MatrixXd &g,const Eigen::MatrixXd &m)
    {
//        std::cout << std::endl << "predKFE: a" << std::endl
//                  << a <<  std::endl;
//        std::cout << std::endl << "predKFE: g" << std::endl
//                  << g <<  std::endl;
//        std::cout << std::endl << "predKFE: m" << std::endl
//                  << m <<  std::endl;
        auto ret = kfe.predict(a,g,m);
        return /*py::make_tuple(*/ret.first/*,ret.second)*/;
    }
//#TODO
//    /*py::tuple*/Eigen::MatrixXd predKFE(const Eigen::MatrixXd &a,const Eigen::MatrixXd &m,const Eigen::MatrixXd &u,const Eigen::MatrixXd &b)
//    {
//        std::cout << std::endl << "predKFE: a" << std::endl
//                  << a <<  std::endl;
//        std::cout << std::endl << "predKFE: m" << std::endl
//                  << m <<  std::endl;
//        std::cout << std::endl << "predKFE: u" << std::endl
//                  << u <<  std::endl;
//        std::cout << std::endl << "predKFE: b" << std::endl
//                  << b <<  std::endl;
//        auto ret = kfe.predict(a,m,u,b);
//        std::cout << std::endl << "predKFE" << std::endl;
//        return /*py::make_tuple(*/ret.first/*,ret.second)*/;
//    }

    /*py::tuple*/Eigen::MatrixXd corrKFE(const Eigen::MatrixXd &m,const Eigen::MatrixXd &z,const Eigen::MatrixXd &r)
    {
        //auto ret = kfe.correct(m,z,r);
        return /*py::make_tuple(*//*ret*/kfe.correct(m,z,r).first/*,ret.second)*/;
    }
//#TODO
//    /*py::tuple*/Eigen::MatrixXd corrKFE(const Eigen::MatrixXd &m,const Eigen::MatrixXd &z)
//    {
//        //auto ret = kfe.correct(m,z);
//        return /*py::make_tuple(*//*ret*/kfe.correct(m,z).first/*,ret.second)*/;
//    }
//#TODO
//    /*py::tuple*/Eigen::MatrixXd corrKFE(const Eigen::MatrixXd &z)
//    {
//        //auto ret = kfe.correct(z);
//        return /*py::make_tuple(*//*ret*/kfe.correct(z).first/*,ret.second)*/;
//    }
};

void bind_kf(pybind11::module &m)
{
    py::class_<BindKFE>(m, "BindKFE")
        .def(py::init<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>())
        .def("predict",&BindKFE::predKFE)
        .def("correct",&BindKFE::corrKFE);
}
