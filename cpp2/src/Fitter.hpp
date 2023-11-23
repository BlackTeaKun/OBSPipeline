#pragma once
#include "Misc.hpp"
#include "Eigen/Dense"
#include <vector>

class Fitter{
    public: 
    Fitter(int ndet, const iVec &bands);
    void addData(DataFlow in);
    void reset();
    std::map<int, BeamParams> getParams();

    private:
    int ndet;
    std::map<int, std::vector<Eigen::Matrix<double, 6, 6>>> xTx;
    std::map<int, std::vector<Eigen::Vector<double, 6>>> xTy;
    iVec bands;
};