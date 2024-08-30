#pragma once

#include "track.h"
#include "measurement.h"
#include "gnn.h"
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
class TrackerGNN
{
private:
public:
    static const int tracks_amount = 1000;//#TODO - подобрать нужный тип! //#TODO - зачем static?
    int current_ready_track=0;
    TrackType global_tracks[tracks_amount];
    std::vector<TrackType*> tracks;

    int counter;

    void initTrack(MeasurementType m)
    {
        global_tracks[current_ready_track].initialization(m);
        tracks.push_back(&global_tracks[current_ready_track]);
        current_ready_track++;if(current_ready_track==tracks_amount)current_ready_track=0;//#TODO - сделать красиво!
    }
    //void updateTrack(/*...*/){}//#TODO
    //void killTrack(/*...*/){}//#TODO
public:
    TrackerGNN():counter(0){}
    std::vector<TrackType*>& step(std::vector<MeasurementType> ms,double dt)
    {
        counter++;
        Eigen::MatrixXd association_matrix = Association::GlobalNearestNeighbor<Eigen::MatrixXd,TrackType,MeasurementType>()(tracks,ms);
        std::vector<unsigned int> vacant_track_numbers;
        std::vector<unsigned int> vacant_measurement_numbers;
        //find pair-numbers
        std::map<int,int> pairs = Association::Auction<Eigen::MatrixXd>()(association_matrix);
        //find vacant-measurement-numbers
        for(int i=0;i<ms.size();i++)
        {
            bool b_ready = false;
            for(auto j : pairs)
                if(i==j.second)
                    b_ready = true;
            if(!b_ready)
                vacant_measurement_numbers.push_back(i);
        }
        //find vacant-tracks-numbers
        for(int i=0;i<tracks.size();i++)
        {
            bool b_ready = false;
            for(auto j : pairs)
                if(i==j.first)
                    b_ready = true;
            if(!b_ready)
                vacant_track_numbers.push_back(i);
        }
        //update tracks
        for(auto i : pairs)
        {
            auto a = tracks.at(i.first)->step(ms.at(i.second));
            //std::cout << "UPDATE: [" << i.first << "]<-(" << i.second << "): " << Utils::transpose(a.first) << std::endl;
        }

        //pass-update tracks
        for(int i=0;i<vacant_track_numbers.size();i++)
        {
            auto a = tracks[vacant_track_numbers.at(i)]->step(tracks[vacant_track_numbers.at(i)]->getTimePoint()+dt);
            //std::cout << "PASS: [" << vacant_track_numbers.at(i) << "]: " << Utils::transpose(a.first) << std::endl;
        }
        //new tracks
        for(int i=0;i<vacant_measurement_numbers.size();i++)
        {
            //std::cout << "NEW_TRACK: from(" << vacant_measurement_numbers.at(i) << ")" << std::endl;
            initTrack(ms.at(vacant_measurement_numbers.at(i)));
        }
        return tracks;
    }
};

}
