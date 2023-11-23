#include "Driver.hpp"
#include "tbb/tbb.h"

DataFlow Driver::reckon(DataFlow in){
    int ndet  = in.fp.r->size();
    int nsamp = in.ces->ra.size();
    in.theta = std::make_shared<Arr2D<double>>(nsamp, ndet);
    in.phi   = std::make_shared<Arr2D<double>>(nsamp, ndet);
    in.pa    = std::make_shared<Arr2D<double>>(nsamp, ndet);
    std::vector<Eigen::Quaterniond> qBoresight(nsamp);
    tbb::parallel_for(0, nsamp, 
        [&](int i){
            double ipa    = in.ces->pa[i];
            double itheta = halfpi - in.ces->dec[i];
            double iphi   = in.ces->ra[i];
            Eigen::Quaterniond qPa(Eigen::AngleAxis<double>(-ipa, Eigen::Vector3d::UnitZ()));
            Eigen::Quaterniond qDec(Eigen::AngleAxis<double>(itheta, Eigen::Vector3d::UnitY()));
            Eigen::Quaterniond qRa(Eigen::AngleAxis<double>(iphi, Eigen::Vector3d::UnitZ()));
            qBoresight[i] = qRa * qDec * qPa;
        }
    );
    tbb::parallel_for(0, ndet, 
        [&](int idet){
            double ir     = (*in.fp.r)[idet];
            double itheta = (*in.fp.theta)[idet];
            double ichi   = (*in.fp.chi)[idet];
            Eigen::Quaterniond qZ(0, 0, 0, 1);
            Eigen::Quaterniond qPol(0, 1, 0, 0);
            Eigen::Quaterniond qR(Eigen::AngleAxis<double>(ir, Eigen::Vector3d::UnitY()));
            Eigen::Quaterniond qTheta(Eigen::AngleAxis<double>(itheta, Eigen::Vector3d::UnitZ()));
            Eigen::Quaterniond qDetRot = qTheta*qR;

            for(int isamp = 0; isamp < nsamp; ++isamp){
                Eigen::Quaterniond qTotal = qBoresight[isamp] * qDetRot;
                Eigen::Vector3d qV = (qTotal * qZ * qTotal.conjugate()).vec();
                (*in.theta)(isamp, idet) = std::acos(qV.z());
                (*in.phi)(isamp, idet) = std::atan2(qV.y(), qV.x());

                Eigen::Quaterniond toXY = Eigen::Quaterniond::FromTwoVectors(qV, Eigen::Vector3d(qV.x(), qV.y(), 0));
                Eigen::Quaterniond toX = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d(qV.x(), qV.y(), 0), Eigen::Vector3d::UnitX());
                qTotal = toX * toXY * qTotal;
                Eigen::Vector3d vP = (qTotal * qPol * qTotal.conjugate()).vec();
                (*in.pa)(isamp, idet) = std::atan2(-vP.y(), -vP.z()) + ichi;
            }
        }
    );
    // in.theta = theta;
    // in.phi   = phi;
    // in.pa    = pa;
    return in;
}