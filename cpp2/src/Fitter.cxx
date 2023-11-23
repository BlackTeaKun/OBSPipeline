#include "Fitter.hpp"
#include "Eigen/Dense"
#include "tbb/tbb.h"
#include <iostream>

Fitter::Fitter(int subndet, const iVec &_bands): ndet(subndet), bands(_bands){
    for(auto v : bands){
        xTx[v] = std::vector<Eigen::Matrix<double, 6, 6>>(ndet);
        xTy[v] = std::vector<Eigen::Vector<double, 6>>(ndet);
    }
    reset();
}

void Fitter::addData(DataFlow in){
    int nsamp = in.pa->rows();
    for(auto v : bands){
        auto diffPtr = in.toddiff[v];
        auto sigma = in.fp.beamsize[v];
        auto sigma2 = sigma*sigma;
        tbb::parallel_for(0, ndet, [&](int idet){
        // for(int idet = 0; idet < ndet; ++idet){
            for(int isamp = 0; isamp < nsamp; ++isamp){
                int ipix = (*in.idx)(isamp, idet);
                // g
                double g_temp = (*in.temp[v].T)[ipix];

                double cospa = std::cos((*in.pa)(isamp, idet));
                double sinpa = std::sin((*in.pa)(isamp, idet));
                // x
                double x_temp = cospa * (*in.temp[v].T_t)[ipix] + sinpa * (*in.temp[v].T_p)[ipix];
                // y 
                double y_temp = -sinpa * (*in.temp[v].T_t)[ipix] + cospa * (*in.temp[v].T_p)[ipix];
                // s
                double s_temp = ((*in.temp[v].T_tt)[ipix] + (*in.temp[v].T_pp)[ipix]) * sigma;

                double cos2pa = std::cos(2 * (*in.pa)(isamp, idet));
                double sin2pa = std::sin(2 * (*in.pa)(isamp, idet));
                double cot    = 1/std::tan((*in.theta)(isamp, idet));
                double dtt = (*in.temp[v].T_tt)[ipix];
                double dpp = (*in.temp[v].T_pp)[ipix];
                double dtp = (*in.temp[v].T_tp)[ipix];
                double dt  = (*in.temp[v].T_t)[ipix];
                double dp  = (*in.temp[v].T_p)[ipix];
                // p
                double p_temp = sigma2*(sin2pa*(cot*dp + dtp) + cos2pa/2* (dtt-cot*dt - dpp));
                // c
                double c_temp = sigma2*((dtp - cot*dp) + sin2pa/2 * (cot*dt - dtt + dpp));

                Eigen::VectorXd x(6), y(2);
                x << g_temp, x_temp, y_temp, s_temp, p_temp, c_temp;
                xTx[v][idet] += x * x.transpose();
                xTy[v][idet] += x * (*diffPtr)(isamp, idet);
            }
        });
    }
}

void Fitter::reset(){
    for(auto v : bands)
        for(int i = 0; i < ndet; ++i){
            xTx[v][i].setZero();
            xTy[v][i].setZero();
        }
}

std::map<int, BeamParams> Fitter::getParams(){
    std::map<int, BeamParams> res;
    for(auto v : bands){
        res[v] = BeamParams(ndet);
        for(int i = 0; i < ndet; ++i){
            Eigen::VectorXd params = xTx[v][i].inverse() * xTy[v][i];
            (*res[v].dg)[i] = params(0);
            (*res[v].dx)[i] = params(1);
            (*res[v].dy)[i] = params(2);
            (*res[v].ds)[i] = params(3);
            (*res[v].dp)[i] = params(4);
            (*res[v].dc)[i] = params(5);
        }
    }
    reset();
    return res;
}