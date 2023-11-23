#pragma once
#include "Misc.hpp"

class MapMaker{
    friend class Driver;
    public: 
    MapMaker(int npix, const iVec &bands);
    void addScan(DataFlow);
    private:
    std::vector<long> bands;
    dVec hit, c2p, csp, s2p;
    std::map<int, dVec> tsum, cpd, spd;
};