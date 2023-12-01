#include "Driver.hpp"
#include "Misc.hpp"

#include "tbb/tick_count.h"
#include "tbb/tbb.h"

#include <memory>
#include <limits>
#include <functional>
#include <algorithm>

#include <iostream>
#include <ios>
#include <iomanip>

#include <Eigen/Dense>
#include <Eigen/Geometry>


Driver::Driver(FocalPlane _fp, std::map<int, Maps> _maps, Config _cfg):
fp(_fp), maps(_maps), deriv_maps(), cfg(_cfg),nscansets(0), total_nfit(0), edgeInitilized(false),
inputNode(g, _cfg.nproc, std::bind(&Driver::reckon, this, std::placeholders::_1)),
samplingNode(g, _cfg.nproc, std::bind(&Driver::sampling, this, std::placeholders::_1)),
leakageNode(g, _cfg.nproc, std::bind(&Driver::addleakage, this, std::placeholders::_1)),
leakageByPassNode(g, _cfg.nproc, std::bind(&Driver::addleakage_bypass, this, std::placeholders::_1)),
joinNode(g),
mapMakerPtrBuffer(g),
mapMakingNode(g, tbb::flow::unlimited, std::bind(&Driver::mapmaking, this, std::placeholders::_1)),
bcastNode(g),
fittingNode(g, _cfg.nproc, std::bind(&Driver::fitting, this, std::placeholders::_1)),
crosstalkNode(g, _cfg.nproc, std::bind(&Driver::crosstalk, this, std::placeholders::_1))
{
    for(auto &kv : maps){
        bands.push_back(kv.first);
    }
    // split detectors;
    int ndet = fp.r->size();
    int idet = 0;
    assert(ndet % _cfg.nproc == 0);
    int ndet_per_proc = ndet / _cfg.nproc;
    sub_fp = std::make_shared<std::vector<FocalPlane>>(_cfg.nproc);
    for(int i = 0; i < cfg.nproc; ++i){
        (*sub_fp)[i].r     = std::make_shared<dVec>(ndet_per_proc);
        (*sub_fp)[i].theta = std::make_shared<dVec>(ndet_per_proc);
        (*sub_fp)[i].chi   = std::make_shared<dVec>(ndet_per_proc);
        FocalPlane &ifocal = (*sub_fp)[i];
        for(int j = 0; j < ndet_per_proc; ++j){
            (*ifocal.r)[j] = (*fp.r)[idet];
            (*ifocal.theta)[j] = (*fp.theta)[idet];
            (*ifocal.chi)[j] = (*fp.chi)[idet];
            ++idet;
        }
    }

    for(int i = 0; i < _cfg.nmapmaking; ++i){
        auto ptr = std::make_shared<MapMaker>(_cfg.npix, bands);
        vMapMaker.push_back(ptr);
        mapMakerPtrBuffer.try_put(ptr);
    }

    for(auto v : bands){
        fittedRes[v] = std::vector<std::vector<BeamParams>>(_cfg.nproc);
    }
}

void Driver::addScan(const std::vector<std::shared_ptr<CES>> &cess){
    if( !edgeInitilized ) make_all_edges();
    for(int ices = 0; ices < cess.size(); ++ices){
        for(int i = 0; i < cfg.nproc; ++i){
            FocalPlane &ifocal = (*sub_fp)[i];
            DataFlow in;
            in.ces   = cess[ices];
            in.fp    = ifocal;
            in.iproc = i;
            in.temp  = deriv_maps;
            in.doFit = false;
            in.iscanset = nscansets;

            if(fitScanNo && std::find(fitScanNo->begin(), fitScanNo->end(), nscansets) != fitScanNo->end() && ices == 0){
                in.doFit = true;
            }
            inputNode.try_put(in);
        }
    }

    if(fitScanNo && std::find(fitScanNo->begin(), fitScanNo->end(), nscansets) != fitScanNo->end()){
        total_nfit += 1;
    }
    ++nscansets;
    if(nscansets % 2 ==0) g.wait_for_all();
}

SkyMaps Driver::getMaps(){
    g.wait_for_all();
    auto c2p = std::make_shared<dVec>(cfg.npix);
    auto s2p = std::make_shared<dVec>(cfg.npix);
    auto csp = std::make_shared<dVec>(cfg.npix);
    auto tsum = std::make_shared<std::map<int, std::shared_ptr<dVec>>>();
    auto cpd = std::make_shared<std::map<int, std::shared_ptr<dVec>>>();
    auto spd = std::make_shared<std::map<int, std::shared_ptr<dVec>>>();
    for(auto v : bands){
        (*tsum)[v] = std::make_shared<dVec>(cfg.npix);
        (*cpd)[v]  = std::make_shared<dVec>(cfg.npix);
        (*spd)[v]  = std::make_shared<dVec>(cfg.npix);
    }
    auto hit = std::make_shared<dVec>(cfg.npix);
    for(int i = 0; i < vMapMaker.size(); ++i){
        tbb::parallel_for(0, cfg.npix, 
            [&](int j){
                (*hit)[j] += vMapMaker[i]->hit[j];
                (*c2p)[j] += vMapMaker[i]->c2p[j];
                (*s2p)[j] += vMapMaker[i]->s2p[j];
                (*csp)[j] += vMapMaker[i]->csp[j];
                for(auto v : bands){
                    (*(*cpd)[v])[j] += vMapMaker[i]->cpd[v][j];
                    (*(*spd)[v])[j] += vMapMaker[i]->spd[v][j];
                    (*(*tsum)[v])[j] += vMapMaker[i]->tsum[v][j];
                }
            }
        );
    }
//solve
    SkyMaps result;
    for(int v : bands){
        auto tmap = std::make_shared<dVec>(cfg.npix);
        tbb::parallel_for(0, cfg.npix, [&](int i){
            (*tmap)[i] = (*(*tsum)[v])[i]/(*hit)[i];
            });

        auto qmap = std::make_shared<dVec>(cfg.npix);
        auto umap = std::make_shared<dVec>(cfg.npix);
        tbb::parallel_for(0, cfg.npix, [&](int i ){
                double det = (*c2p)[i]*(*s2p)[i] - (*csp)[i]*(*csp)[i];
                (*qmap)[i] = ((*s2p)[i]*(*(*cpd)[v])[i] - (*csp)[i]*(*(*spd)[v])[i])/det;
                (*umap)[i] = ((*c2p)[i]*(*(*spd)[v])[i] - (*csp)[i]*(*(*cpd)[v])[i])/det;
            }
        );
        result.T[v] = tmap;
        result.Q[v] = qmap;
        result.U[v] = umap;

        result.tsum[v] = (*tsum)[v];
        result.cpd[v] = (*cpd)[v];
        result.spd[v] = (*spd)[v];
    }
    result.hitmap = hit;
    result.c2p = c2p;
    result.csp = csp;
    result.s2p = s2p;
    return result;
}

std::shared_ptr<MapMaker> Driver::mapmaking(std::tuple<DataFlow, std::shared_ptr<MapMaker>> in){
    std::get<1>(in)->addScan(std::get<0>(in));
    return std::get<1>(in);
}

void Driver::fitting(DataFlow in){
    if (in.doFit){
        auto tmp = vFitter[in.iproc]->getParams();
        for(auto v : bands){
            fittedRes[v][in.iproc].push_back(tmp[v]);
        }
    }
    vFitter[in.iproc]->addData(in);
}

void Driver::addBeamSysParams(std::map<int, BeamParams> beampara, std::map<int, DerivMaps> deriv){
    cfg.doSysmatic = true;
    fp.beamsys = beampara;
    deriv_maps = deriv;


    //split to subfp
    int ndet = fp.r->size();
    int ndet_per_proc = ndet / cfg.nproc;
    for(auto v : bands){
        int idet = 0;
        for(int i = 0; i < cfg.nproc; ++i){
            FocalPlane &ifocal = (*sub_fp)[i];
            ifocal.beamsys[v].dg = std::make_shared<dVec>(ndet_per_proc);
            ifocal.beamsys[v].dx = std::make_shared<dVec>(ndet_per_proc);
            ifocal.beamsys[v].dy = std::make_shared<dVec>(ndet_per_proc);
            ifocal.beamsys[v].ds = std::make_shared<dVec>(ndet_per_proc);
            ifocal.beamsys[v].dp = std::make_shared<dVec>(ndet_per_proc);
            ifocal.beamsys[v].dc = std::make_shared<dVec>(ndet_per_proc);
            ifocal.beamsize = fp.beamsize;

            for(int j = 0; j < ndet_per_proc; ++j){
                (*ifocal.beamsys[v].dg)[j] = (*fp.beamsys[v].dg)[idet];
                (*ifocal.beamsys[v].dx)[j] = (*fp.beamsys[v].dx)[idet];
                (*ifocal.beamsys[v].dy)[j] = (*fp.beamsys[v].dy)[idet];
                (*ifocal.beamsys[v].ds)[j] = (*fp.beamsys[v].ds)[idet];
                (*ifocal.beamsys[v].dp)[j] = (*fp.beamsys[v].dp)[idet];
                (*ifocal.beamsys[v].dc)[j] = (*fp.beamsys[v].dc)[idet];
                ++idet;
            }
        }
    }
}

void Driver::addFittingScansets(std::shared_ptr<iVec> &scan_no){
    cfg.fitSysmatic = true;
    fitScanNo = scan_no;
}

void Driver::addFittingTemplate(std::map<int, DerivMaps> temp){
    temp_maps = temp;
    cfg.fitSysmatic = true;

    int ndet = fp.r->size();
    int idet = 0;
    int ndet_per_proc = ndet / cfg.nproc;
    for(int i = 0; i < cfg.nproc; ++i){
        vFitter.push_back(std::make_shared<Fitter>(ndet_per_proc, bands));
    }
}

std::map<int, std::vector<BeamParams>> Driver::getFittedParams(){
    g.wait_for_all();
    int ndet = fp.r->size();
    int ndet_per_proc = ndet / cfg.nproc;
    for(int i = 0; i < cfg.nproc; ++i){
        auto tmp = vFitter[i]->getParams();
        for(auto v : bands)
            fittedRes[v][i].push_back(tmp[v]);
    }
    total_nfit += 1;
    std::map<int, std::vector<BeamParams>> result;
    for(auto v: bands){
        result[v] = std::vector<BeamParams>(total_nfit);

        for(int i = 0; i < total_nfit; ++i){
            result[v][i] = BeamParams(ndet);
        }

        for(int i = 0; i < cfg.nproc; ++i){
            for(int j = 0; j < total_nfit; ++j){
                for(int idet = 0; idet < ndet_per_proc; ++idet){
                    result[v][j].dg->at(i*ndet_per_proc+idet) = fittedRes[v][i][j].dg->at(idet);
                    result[v][j].dx->at(i*ndet_per_proc+idet) = fittedRes[v][i][j].dx->at(idet);
                    result[v][j].dy->at(i*ndet_per_proc+idet) = fittedRes[v][i][j].dy->at(idet);
                    result[v][j].ds->at(i*ndet_per_proc+idet) = fittedRes[v][i][j].ds->at(idet);
                    result[v][j].dp->at(i*ndet_per_proc+idet) = fittedRes[v][i][j].dp->at(idet);
                    result[v][j].dc->at(i*ndet_per_proc+idet) = fittedRes[v][i][j].dc->at(idet);
                }
            }
        }
    }
    return result;
}

void Driver::addFittedParams(std::map<int, std::vector<BeamParams>> in){
    cfg.doDeproj = true;
    fittedParams = in;
}

void Driver::make_all_edges(){
    tbb::flow::make_edge(inputNode, samplingNode);
    //tbb::flow::make_edge(samplingNode, bcastNode);
    tbb::flow::make_edge(bcastNode, tbb::flow::input_port<0>(joinNode));
    tbb::flow::make_edge(mapMakerPtrBuffer, tbb::flow::input_port<1>(joinNode));
    tbb::flow::make_edge(joinNode, mapMakingNode);
    tbb::flow::make_edge(mapMakingNode, mapMakerPtrBuffer);

    tbb::flow::function_node<DataFlow, DataFlow> *lastTODNode = &samplingNode;
    if(cfg.doSysmatic){ // beam systematic
        if(cfg.doDeproj){
            tbb::flow::make_edge(samplingNode, leakageByPassNode);
            tbb::flow::make_edge(leakageByPassNode, bcastNode);
            lastTODNode = &leakageByPassNode;
        }
        else{
            tbb::flow::make_edge(samplingNode, leakageNode);
            lastTODNode = &leakageNode;
        }
    }
    if(cfg.doCrosstalk){
        tbb::flow::make_edge(*lastTODNode, crosstalkNode);
        lastTODNode = &crosstalkNode;
    }
    tbb::flow::make_edge(*lastTODNode, bcastNode);
    if(cfg.fitSysmatic){
        tbb::flow::make_edge(bcastNode, fittingNode);
    }
    edgeInitilized = true;
}


void Driver::addMixingMatrix(const std::vector<std::shared_ptr<Arr2D<double>>> &mixmtr){
    mixingMtr = mixmtr;
}