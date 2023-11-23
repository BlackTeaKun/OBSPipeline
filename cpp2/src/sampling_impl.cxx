#include "Driver.hpp"
#include "tbb/tbb.h"
#include "healpix_cxx/healpix_map.h"
#include "healpix_cxx/pointing.h"

DataFlow Driver::sampling(DataFlow in){
    int ndet  = in.pa->cols();
    int nsamp = in.pa->rows();
    for(auto it = maps.begin(); it != maps.end(); ++it){
        in.todsum[it->first] = std::make_shared<Arr2D<double>>(nsamp, ndet);
        in.toddiff[it->first] = std::make_shared<Arr2D<double>>(nsamp, ndet);;
    }
    in.idx = std::make_shared<Arr2D<int>>(nsamp, ndet);

    Healpix_Map<int> base;
    base.SetNside(cfg.nside, RING);

    tbb::parallel_for(0, ndet, 
        [&](int i){
            for (int j = 0; j < nsamp; ++j) {
                double itheta = (*in.theta)(j, i);
                double iphi   = (*in.phi)(j, i);
                int ipix = base.ang2pix({itheta, iphi});
                (*in.idx)(j, i) = ipix;
                for(auto it = maps.begin(); it != maps.end(); ++it){
                    (*in.todsum[it->first])(j, i) = (*it->second.T)[ipix] * 2;

                    double q = (*it->second.Q)[ipix];
                    double u = (*it->second.U)[ipix];
                    double twopsi = (*in.pa)(j, i) * 2;
                    (*in.toddiff[it->first])(j, i) = (q*std::cos(twopsi) + u*std::sin(twopsi)) * 2;
                }
            }
        }
    );
    return in;
}