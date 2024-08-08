#pragma once
#include "track.h"

namespace Association
{

template<class M, class TrackType, class MeasurementType>
class GlobalNearestNeighbor
{
public:
    M operator()(std::vector<TrackType*>& tracks, std::vector<MeasurementType>& measurements)
    {
        // == zero matrix create ==
        int tracks_size = tracks.size();
        int measurements_size = measurements.size();

        if(tracks_size==0 || measurements_size==0)
            tracks_size=measurements_size=0;

        M d2(tracks_size,measurements_size);
        d2.setZero();

        M a(tracks_size,measurements_size);
        a.setZero();

        // == cycle ==
        for(int i=0;i<tracks.size();i++)
        {
            for(int j=0;j<measurements.size();j++)
            {
                double dt = measurements.at(j).timepoint() - tracks.at(i)->getTimePoint();
                auto zp_S = tracks.at(i)->getMeasurementPredictData(dt);
                auto G = 50000.;//#ZAGL - //tracks.at(i)->getGate();
                //auto Pd = 0.;
                //auto beta =
                //auto G0 = 2*std::ln(Pd/((1-Pd)*std::pow((2*M_PI),m/2)*beta*std::sqrt(Utils::zp_S.determinant())));
                M zp = zp_S.first;
                M S = zp_S.second;
                M z = measurements.at(j).get();
                M y = z - zp;
                d2(i,j) = (Utils::transpose(y)*Utils::inverse(S)*y).determinant();
                a(i,j) = G-d2(i,j);
            }
        }
        //std::cout << "GlobalNearestNeighbor: [result] a[" << a.rows() << "," << a.cols() << "]" << std::endl << a << std::endl;
        return a;
    }
};

template<class M>
class Auction
{
private:
public:
    bool is_ass_meas(int meas_num, std::map<int,int> map)
    {
        for(auto const& i : map)
            if(i.second==meas_num)
                return true;
        return false;
    }
    std::map<int,int> operator()(M assignment_matrix)
    {
        M m = assignment_matrix;
        double e = 0.15;//1./m.rows();//#TODO -

        std::map<int,int> pairs;
        std::map<int,double> prices;
        for(int i=0;i<m.rows();i++)
            prices.insert({i,0.});

        int counter = 0;

        bool bj = true;
        while(bj)
        {
            bj = false;
            for(int j=0;j<m.cols();j++)
            {
                if(is_ass_meas(j,pairs))
                    continue;
                counter++;

                double max_ap = 0.;//-1000000000.;
                int max_i = 0;
                bool b_max = false;

                double nex_ap = 0.;//-1000000000.;
                int nex_i = 0;
                bool b_nex = false;

                for(int i=0;i<m.rows();i++)
                {
                    double ap = m(i,j) - prices.at(i);

                    if(ap>=max_ap)
                    {
                        nex_ap = max_ap;
                        nex_i = max_i;
                        b_nex = b_max;

                        max_ap = ap;
                        max_i = i;
                        b_max = true;
                    }
                    else if(ap>=nex_ap)
                    {
                        nex_ap = ap;
                        nex_i = i;
                        b_nex = true;
                    }
                }

                if(!b_max)
                    continue;

                bj = true;

                if(pairs.count(max_i))
                    pairs.erase(max_i);
                pairs.insert({max_i,j});

                prices[max_i] = prices[max_i] + (max_ap-nex_ap) + e;
            }
        }

        //std::cout << "Auction: [" << counter << "] pairs: ";for(auto i : pairs)std::cout  << "[" << i.first << "]-(" << i.second << ") "; std::cout << std::endl;
        return pairs;
    }
};
}
