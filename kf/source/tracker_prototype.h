#pragma once

#include "track.h"
#include "measurement.h"
#include "gnn_prototype.h"
#include "kf.h"
#include "utils.h"
#include <cmath>
#include <typeinfo>
#include <algorithm>

namespace Tracker
{

template<class M,
         class EstimatorInitializatorType,
         class EstimatorType,
         class TrackType,
         class MeasurementType>
class Tracker_prototype
{
public:
    static const int tracks_amount = 1000;
    int current_ready_track=0;
    TrackType global_tracks[tracks_amount];
    std::vector<TrackType*> tracks;

    void initTrack(TrackType& t, MeasurementType m)
    {
        t.initialization(m);
        tracks.push_back(&t);
    }
public:
    Tracker_prototype(){}
    std::vector<M> step(std::vector<MeasurementType> ms)
    {

        //----------------------------------------------------------------------
        Eigen::MatrixXd association_matrix = Association::GNN_prototype_2<Eigen::MatrixXd,TrackType,MeasurementType>()(tracks,ms);
        std::vector<std::pair<int,int>> result_pairs = Association::Auction_prototype<Eigen::MatrixXd>()(association_matrix/*,vec_prices*/);
        //----------------------------------------------------------------------
        std::vector<unsigned int> ready_track_numbers;
        std::vector<unsigned int> ready_measurement_numbers;
        for(int i=0;i<result_pairs.size();i++)
        {

            tracks[result_pairs.at(i).first]->step(ms.at(result_pairs.at(i).second));
            ready_track_numbers.push_back(result_pairs.at(i).first);
            ready_measurement_numbers.push_back(result_pairs.at(i).second);
        }
        //----------------------------------------------------------------------
        for(int i=0;i<ms.size();i++)
        {
            bool b_ready = false;
            for(int j=0;j<ready_measurement_numbers.size();j++)
            {
                if(i==ready_measurement_numbers.at(j))
                    b_ready = true;
            }
            if(!b_ready)
            {
                initTrack(global_tracks[current_ready_track],ms.at(i));
                current_ready_track++;if(current_ready_track==tracks_amount)current_ready_track=0;
            }
        }

        return std::vector<M>();//#ZAGL
    }
};

}
