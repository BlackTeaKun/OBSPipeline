#pragma once
#include <vector>
#include <map>
#include <memory>
#include <iostream>
#include <tbb/flow_graph.h>
#include <tbb/enumerable_thread_specific.h>
#include "Misc.hpp"
#include "MapMaker.hpp"
#include "Fitter.hpp"

class Driver{
    public:
    Driver(FocalPlane, std::map<int, Maps>, Config);
    void addBeamSysParams(std::map<int, BeamParams>, std::map<int, DerivMaps>);
    void addFittingScansets(std::shared_ptr<std::vector<long>> &scan_no);
    void addFittingTemplate(std::map<int, DerivMaps>);
    void addFittedParams(std::map<int, std::vector<BeamParams>>);
    void addScan(const std::vector<std::shared_ptr<CES>> &);

    SkyMaps getMaps();
    std::map<int, std::vector<BeamParams>> getFittedParams();
    void wait(){g.wait_for_all();}

    void addMixingMatrix(const std::vector<std::shared_ptr<Arr2D<double>>> &);

    ~Driver(){
        wait();
    }
    
    private:
    DataFlow reckon(DataFlow);
    DataFlow sampling(DataFlow);
    DataFlow addleakage(DataFlow);
    DataFlow addleakage_bypass(DataFlow);
    void     fitting(DataFlow);
    DataFlow crosstalk(DataFlow);
    std::shared_ptr<MapMaker> mapmaking(std::tuple<DataFlow, std::shared_ptr<MapMaker>>);

    void     make_all_edges();
    bool     edgeInitilized;

    tbb::flow::graph g;
    tbb::flow::function_node<DataFlow, DataFlow> inputNode;
    tbb::flow::function_node<DataFlow, DataFlow> samplingNode;
    tbb::flow::function_node<DataFlow, DataFlow> leakageNode;
    tbb::flow::function_node<DataFlow, DataFlow> leakageByPassNode;
    tbb::flow::function_node<DataFlow, DataFlow> crosstalkNode;
    tbb::flow::join_node<std::tuple<DataFlow, std::shared_ptr<MapMaker>>> joinNode;
    tbb::flow::buffer_node<std::shared_ptr<MapMaker>> mapMakerPtrBuffer;
    tbb::flow::broadcast_node<DataFlow> bcastNode;
    tbb::flow::function_node<DataFlow> fittingNode;
    
    tbb::flow::function_node<
        std::tuple<DataFlow, std::shared_ptr<MapMaker>>,
        std::shared_ptr<MapMaker>
    > mapMakingNode;
    std::vector<std::shared_ptr<MapMaker>> vMapMaker;
    std::vector<std::shared_ptr<Fitter>>   vFitter;
    std::map<int, std::vector<std::vector<BeamParams>>>  fittedRes;

    FocalPlane fp;
    std::map<int, Maps>            maps;
    std::map<int, DerivMaps> deriv_maps;
    std::map<int, DerivMaps>  temp_maps;
    iVec bands;
    std::shared_ptr<iVec> fitScanNo;

    std::vector<std::shared_ptr<Arr2D<double>>> mixingMtr;

    std::shared_ptr<std::vector<FocalPlane>> sub_fp;
    std::map<int, std::vector<BeamParams>> fittedParams;

    Config cfg;
    int nscansets, total_nfit;
};
