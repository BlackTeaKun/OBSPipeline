#include "Driver.hpp"
#include "tbb/tbb.h"
#include <algorithm>
DataFlow Driver::addleakage_bypass(DataFlow in){
    int iscanset = in.iscanset;
    int paraidx = std::lower_bound(fitScanNo->begin(), fitScanNo->end(), iscanset) - fitScanNo->begin();
    int ndet  = in.pa->cols();
    int nsamp = in.pa->rows();
    int offset = in.iproc * ndet;
    for(auto v : bands){
        auto diffPtr = in.toddiff[v];
        auto sigma = fp.beamsize[v];
        auto sigma2 = sigma*sigma;
        tbb::parallel_for(0, ndet, [&](int idet){
        // for(int idet = 0; idet < ndet; ++idet){
            for(int isamp = 0; isamp < nsamp; ++isamp){
                int ipix = (*in.idx)(isamp, idet);
                // g
                double g_temp = (*maps[v].T)[ipix];

                double cospa = std::cos((*in.pa)(isamp, idet));
                double sinpa = std::sin((*in.pa)(isamp, idet));
                // x
                double x_temp = cospa * (*deriv_maps[v].T_t)[ipix] + sinpa * (*deriv_maps[v].T_p)[ipix];
                // y 
                double y_temp = -sinpa * (*deriv_maps[v].T_t)[ipix] + cospa * (*deriv_maps[v].T_p)[ipix];
                // s
                double s_temp = ((*deriv_maps[v].T_tt)[ipix] + (*deriv_maps[v].T_pp)[ipix]) * sigma;

                double cos2pa = std::cos(2 * (*in.pa)(isamp, idet));
                double sin2pa = std::sin(2 * (*in.pa)(isamp, idet));
                double cot    = 1/std::tan((*in.theta)(isamp, idet));
                double dtt = (*deriv_maps[v].T_tt)[ipix];
                double dpp = (*deriv_maps[v].T_pp)[ipix];
                double dtp = (*deriv_maps[v].T_tp)[ipix];
                double dt  = (*deriv_maps[v].T_t)[ipix];
                double dp  = (*deriv_maps[v].T_p)[ipix];
                // p
                double p_temp = sigma2*(sin2pa*(cot*dp + dtp) + cos2pa/2* (dtt-cot*dt - dpp));
                // c
                double c_temp = sigma2*((dtp - cot*dp) + sin2pa/2 * (cot*dt - dtt + dpp));

                double total = ((*in.fp.beamsys[v].dg)[idet] - (*fittedParams[v][paraidx].dg)[offset+idet]) * g_temp +
                               ((*in.fp.beamsys[v].dx)[idet] - (*fittedParams[v][paraidx].dx)[offset+idet]) * x_temp +
                               ((*in.fp.beamsys[v].dy)[idet] - (*fittedParams[v][paraidx].dy)[offset+idet]) * y_temp +
                               ((*in.fp.beamsys[v].ds)[idet] - (*fittedParams[v][paraidx].ds)[offset+idet]) * s_temp +
                               ((*in.fp.beamsys[v].dp)[idet] - (*fittedParams[v][paraidx].dp)[offset+idet]) * p_temp +
                               ((*in.fp.beamsys[v].dc)[idet] - (*fittedParams[v][paraidx].dc)[offset+idet]) * c_temp;
                (*diffPtr)(isamp, idet) += total;
            }
        });
    }
    return in;
}