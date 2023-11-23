#pragma once

#include <vector>
#include <memory>
#include <map>

#include "Eigen/Dense"

const double halfpi = 1.5707963267948966;

using dVec = std::vector<double>;
using iVec = std::vector<long>;
using VWrapper = Eigen::Map<Eigen::VectorXd>;

template<class T>
struct Arr2D{
    Arr2D(int nsamp, int ndet):m_cols(ndet), m_rows(nsamp), data(new T[ndet*nsamp]){ }
    T &operator()(int isamp, int idet){
        return data[idet*m_rows + isamp];
    }
    Arr2D(Arr2D &) = delete;
    Arr2D &operator=(const Arr2D &rhs) = delete;
    Arr2D(Arr2D &&) = delete;
    ~Arr2D(){delete[] data;}

    T rows(){return this->m_rows;}
    T cols(){return this->m_cols;}

    T* data;
    int m_rows, m_cols;
};

struct BeamParams {
    BeamParams() = default;
    BeamParams(int n) : dg(std::make_shared<dVec>(n)),
                        dx(std::make_shared<dVec>(n)),
                        dy(std::make_shared<dVec>(n)),
                        ds(std::make_shared<dVec>(n)),
                        dp(std::make_shared<dVec>(n)),
                        dc(std::make_shared<dVec>(n)) {}
    std::shared_ptr<dVec> dg, dx, dy, ds, dp, dc;
};

struct Maps{
    std::shared_ptr<VWrapper> T, Q, U;
};

struct DerivMaps{
    std::shared_ptr<VWrapper> T, T_t, T_p, T_tt, T_tp, T_pp;
};

struct CES{
    dVec ra, dec, pa;
};

struct FocalPlane{
    std::shared_ptr<dVec>     r, theta, chi;
    std::map<int, double>     beamsize;
    std::map<int, BeamParams> beamsys;
};

struct Config{
    bool doSysmatic;
    bool fitSysmatic;
    int nside, npix;
    int nproc, nmapmaking;
};

struct DataFlow{
    int iproc, iscanset;
    std::shared_ptr<CES> ces;
    std::shared_ptr<Arr2D<double>> theta, phi, pa;
    std::shared_ptr<Arr2D<int>> idx;
    std::map<int, std::shared_ptr<Arr2D<double>>> todsum, toddiff;
    FocalPlane fp;

    // fitting;
    std::map<int, DerivMaps> temp;
    bool doFit;
};

struct SkyMaps{
    std::shared_ptr<dVec> hitmap;
    std::map<int, std::shared_ptr<dVec>> T, Q, U;

    // useful for MPI 
    std::shared_ptr<dVec> c2p, s2p, csp;
    std::map<int, std::shared_ptr<dVec>> tsum, cpd, spd;
};

