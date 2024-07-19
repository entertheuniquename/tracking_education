#pragma once

#include "track.h"

namespace Association
{
template<class M>
class GNN_prototype
{
public:
    M operator()(std::vector<M> zs, std::vector<M> zps, std::vector<M> Ss)
    {
        M d2(zps.size(),zs.size());
        d2.setZero();
        M dG2(zps.size(),zs.size());
        dG2.setZero();
        for(int i=0;i<zps.size();i++)
        {
            M zp = zps.at(i);
            M S = Ss.at(i);
            for(int j=0;j<zs.size();j++)
            {
                M z = zs.at(j);
                M y = z - zp;
                d2(i,j) = (Utils::transpose(y)*S*y).determinant();
                dG2(i,j) = d2(i,j)+std::log(S.determinant());
            }
        }
        return dG2;
    }
};
template<class M>
class Auction_prototype
{
public:
    std::vector<std::pair<int,int>> operator()(M m)
    {
        std::vector<std::pair<int,int>> result_pairs;
        for(int i=0;i<m.rows();i++)
        {
            double min = 1.e10;
            bool b_min = false;
            int minj = 0;
            for(int j=0;j<m.cols();j++)
            {
                double x = m(i,j);
                if(x<=min)
                {
                    min=x;
                    minj = j;
                    b_min = true;
                }
            }
            result_pairs.push_back(std::make_pair(i,minj));
        }

        return result_pairs;
    }
};
}
