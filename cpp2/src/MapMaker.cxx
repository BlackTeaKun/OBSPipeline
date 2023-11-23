#include "MapMaker.hpp"
#include <cmath>
#include <iostream>
#include "tbb/tbb.h"

MapMaker::MapMaker(int npix, const iVec &_bands):
bands(_bands),
hit(npix), c2p(npix), csp(npix), s2p(npix){
    for(auto v : bands){
        tsum[v] = dVec(npix);
        cpd[v] = dVec(npix);
        spd[v] = dVec(npix);
    }
}

void MapMaker::addScan(DataFlow in){
    int ndet = in.idx->cols();
    int nsamp = in.idx->rows();
    Eigen::ArrayXXd cos2p(nsamp, ndet);
    Eigen::ArrayXXd sin2p(nsamp, ndet);
    tbb::parallel_for(
        0, ndet, [&](int idet){for(int isamp=0; isamp<nsamp; ++isamp) cos2p(isamp, idet)=std::cos(2*(*in.pa)(isamp, idet));}
    );
    tbb::parallel_for(
        0, ndet, [&](int idet){for(int isamp=0; isamp<nsamp; ++isamp) sin2p(isamp, idet)=std::sin(2*(*in.pa)(isamp, idet));}
    );
    tbb::task_group g;
    for(auto v : bands){
        g.run([&, v](){
            for(int idet = 0; idet < ndet; ++idet){
                for(int isamp = 0; isamp < nsamp; ++isamp){
                    int ipix = (*in.idx)(isamp, idet);
                    double ic2p = cos2p(isamp, idet);
                    double is2p = sin2p(isamp, idet);
                    cpd[v][ipix] += ic2p*(*in.toddiff[v])(isamp, idet)/2;
                    spd[v][ipix] += is2p*(*in.toddiff[v])(isamp, idet)/2;
                    tsum[v][ipix] += (*in.todsum[v])(isamp, idet);
                }
            }
        });
    }
    for(int idet = 0; idet < ndet; ++idet){
        for(int isamp = 0; isamp < nsamp; ++isamp){
            int ipix = (*in.idx)(isamp, idet);
            hit[ipix] += 2;
            double ic2p = cos2p(isamp, idet);
            double is2p = sin2p(isamp, idet);
            csp[ipix] += ic2p*is2p;
            c2p[ipix] += ic2p*ic2p;
            s2p[ipix] += is2p*is2p;
        }
    }
    g.wait();
}