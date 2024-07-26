#include "bind_gnn_prototype.h"

namespace py = pybind11;

void bind_gnn_prototype(pybind11::module &m)
{
    m.def("BindGNN_prototype", [](std::vector<Eigen::MatrixXd> zs,
                                  std::vector<Eigen::MatrixXd> zps,
                                  std::vector<Eigen::MatrixXd> Ss)
    {
        Association::GNN_prototype<Eigen::MatrixXd> gnn;
        return gnn(zs,zps,Ss);
    });
    m.def("BindAUCTION_prototype", [](Eigen::MatrixXd m)
    {
        Association::Auction_prototype<Eigen::MatrixXd> auc;
        return auc(m);
    });
}
