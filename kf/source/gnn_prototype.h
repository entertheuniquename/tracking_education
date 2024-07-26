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

template<class M, class TrackType, class MeasurementType>
class GNN_prototype_2
{
public:
    M operator()(std::vector<TrackType*> tracks, std::vector<MeasurementType>& measurements)
    {
        M d2(tracks.size(),measurements.size());
        d2.setZero();
        M dG2(tracks.size(),measurements.size());
        dG2.setZero();
        for(int i=0;i<tracks.size();i++)
        {
            for(int j=0;j<measurements.size();j++)
            {
                double dt = measurements.at(j).timepoint() - tracks.at(i)->getTimePoint();
                auto t0 = tracks.at(i)->getMeasurementPredictData(dt);
                M zp = t0.first;
                M S = t0.second;
                M z = measurements.at(j).get();
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
    std::vector<std::pair<int,int>> operator()(M m/*, std::vector<double> prices*/)
    {
        std::vector<std::pair<int,int>> result_pairs;

        for(int i=0;i<m.rows();i++)
        {
            double min/*max*/ = 1.e10;
            bool b_min/*b_max*/ = false;
            int minj/*maxj*/ = 0;
            for(int j=0;j<m.cols();j++)
            {
                double x = m(i,j)/*-prices.at(i)*/;
                if(x<=min/*x>=max*/)
                {
                    min/*max*/=x;
                    minj/*maxj*/ = j;
                    b_min/*b_max*/ = true;
                }
            }
            result_pairs.push_back(std::make_pair(i,minj));
        }

        return result_pairs;
    }
};
}
