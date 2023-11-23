#define PY_SSIZE_T_CLEAN 
#include <Python.h>
#include "structmember.h"
#include <numpy/arrayobject.h>

#include "Driver.hpp"

#include <algorithm>
#include <iostream>
#include <map>

#include "Eigen/Dense"

struct PyDriver{
    PyObject_HEAD
    Driver *driver;
};

PyObject * PyDriver_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyDriver *self;
    self = (PyDriver *) type->tp_alloc(type, 0);
    if (self != nullptr) {
        self->driver = nullptr;
    }
    return (PyObject *) self;
}

void PyDriver_dealloc(PyDriver *self) {
    delete self->driver;
    self->driver = nullptr;
    Py_TYPE(self)->tp_free((PyObject *) self);
}

int PyDriver_init(PyDriver *self, PyObject *args) {
    PyObject *r, *theta, *chi;
    PyObject *beams;
    PyObject *tquMaps;
    int nside, nproc, nmapmaking;
    if (!PyArg_ParseTuple(args, "iOOOOO", &nside, &tquMaps, &beams, &r, &theta, &chi))
        return -1;
    int npix = nside*nside*12;

    Config cfg;
    cfg.nside = nside;
    cfg.npix  = npix;
    cfg.nproc = 4;
    cfg.nmapmaking = 4;

    int ndet = PyArray_SIZE(r);
    auto vr     = std::make_shared<dVec>(ndet);
    auto vtheta = std::make_shared<dVec>(ndet);
    auto vchi   = std::make_shared<dVec>(ndet);
    for(int i = 0; i < ndet; ++i){
        (*vr)[i]     = *(double*)PyArray_GETPTR1(r, i);
        (*vtheta)[i] = *(double*)PyArray_GETPTR1(theta, i);
        (*vchi)[i]   = *(double*)PyArray_GETPTR1(chi, i);
    }
    PyObject *key, *val;
    Py_ssize_t pos = 0;

    FocalPlane fp;
    fp.r = vr;
    fp.theta = vtheta;
    fp.chi = vchi;
    while(PyDict_Next(beams, &pos, &key, &val)){
        int band = PyLong_AsLong(key);
        double beamsize = PyFloat_AsDouble(val);
        fp.beamsize[band] = beamsize;
    }

    pos = 0;
    std::map<int, Maps> mapsIn;
    while(PyDict_Next(tquMaps, &pos, &key, &val)){
        int band = PyLong_AsLong(key);
        Maps m;
        m.T = std::make_shared<VWrapper>((double*)PyArray_GETPTR2(val, 0, 0), npix);
        m.Q = std::make_shared<VWrapper>((double*)PyArray_GETPTR2(val, 1, 0), npix);
        m.U = std::make_shared<VWrapper>((double*)PyArray_GETPTR2(val, 2, 0), npix);
        mapsIn[band] = m;
    }
    
    self->driver = new Driver(fp, mapsIn, cfg);
    return 0;
}

PyObject * PyDriver_addScan(PyDriver *self, PyObject* args) {
  PyObject *ra, *dec, *pa;
  PyObject *st, *ed;
  int ok = PyArg_ParseTuple(args, "OOOOO", &ra, &dec, &pa, &st, &ed);
  if (!ok){
    return nullptr;
  }
  int nces = PyArray_SIZE(st);
  std::vector<std::shared_ptr<CES>> cess(nces);
  double *pra  = (double *)PyArray_GETPTR1(ra, 0);
  double *pdec = (double *)PyArray_GETPTR1(dec, 0);
  double *ppa  = (double *)PyArray_GETPTR1(pa, 0);
  for(int ices = 0; ices < nces; ++ices){
    int ist = *(int*)PyArray_GETPTR1(st, ices);
    int ied = *(int*)PyArray_GETPTR1(ed, ices);
    
    auto pces = std::make_shared<CES>();
    pces->ra.resize(ied-ist);
    pces->dec.resize(ied-ist);
    pces->pa.resize(ied-ist);
    for(int i = ist; i < ied; ++i){
        pces->ra[i-ist]  = pra[i];
        pces->dec[i-ist] = pdec[i];
        pces->pa[i-ist]  = ppa[i];
    }
    cess[ices] = pces;
  }
  self->driver->addScan(cess);
  Py_RETURN_NONE;
}

PyObject *PyDriver_getMaps(PyDriver *self, PyObject* Py_UNUSED(ignored)){
    auto skymap = self->driver->getMaps();
    npy_intp npix = skymap.hitmap->size();
    std::vector<int> bands;
    for(auto &kv : skymap.T) bands.push_back(kv.first); 
    PyObject *maps_dict = PyDict_New();
    for(auto v : bands){
        PyObject *mapsDict = PyDict_New();

        double *out = new double[3*npix];
        for(int i = 0; i < npix; ++i){
            out[i]        = (*skymap.T[v])[i];
            out[i+npix]   = (*skymap.Q[v])[i];
            out[i+2*npix] = (*skymap.U[v])[i];
        }
        npy_intp dims[2]{3, npix};
        PyObject *mapsArr = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, out);
        PyArray_ENABLEFLAGS((PyArrayObject*)mapsArr, NPY_ARRAY_OWNDATA);
        PyDict_SetItemString(mapsDict, "map", mapsArr);

        double *tsum = new double[npix];
        for(int i = 0; i < npix; ++i){
          tsum[i] = (*skymap.tsum[v])[i];
        }
        PyObject *tsumArr = PyArray_SimpleNewFromData(1, &npix, NPY_FLOAT64, tsum);
        PyArray_ENABLEFLAGS((PyArrayObject*)tsumArr, NPY_ARRAY_OWNDATA);
        PyDict_SetItemString(mapsDict, "tsum", tsumArr);

        double *cpd = new double[npix];
        for(int i = 0; i < npix; ++i){
          cpd[i] = (*skymap.cpd[v])[i];
        }
        PyObject *cpdArr = PyArray_SimpleNewFromData(1, &npix, NPY_FLOAT64, cpd);
        PyArray_ENABLEFLAGS((PyArrayObject*)cpdArr, NPY_ARRAY_OWNDATA);
        PyDict_SetItemString(mapsDict, "cpd", cpdArr);

        double *spd = new double[npix];
        for(int i = 0; i < npix; ++i){
          spd[i] = (*skymap.spd[v])[i];
        }
        PyObject *spdArr = PyArray_SimpleNewFromData(1, &npix, NPY_FLOAT64, spd);
        PyArray_ENABLEFLAGS((PyArrayObject*)spdArr, NPY_ARRAY_OWNDATA);
        PyDict_SetItemString(mapsDict, "spd", spdArr);



        PyObject *iband = PyLong_FromLong(v);
        PyDict_SetItem(maps_dict, iband, mapsDict);
    }
    npy_intp *hit = new npy_intp[npix];
    for(int i = 0; i < npix; ++i){
        hit[i] = (*skymap.hitmap)[i];
    }
    npy_intp hitdims[1]{npix};
    PyObject *hitArr = PyArray_SimpleNewFromData(1, hitdims, NPY_INTP, hit);
    PyArray_ENABLEFLAGS((PyArrayObject*)hitArr, NPY_ARRAY_OWNDATA);
    PyDict_SetItemString(maps_dict, "hitmap", hitArr);

    double *c2p = new double[npix];
    for(int i = 0; i < npix; ++i){
        c2p[i] = (*skymap.c2p)[i];
    }
    PyObject *c2pArr = PyArray_SimpleNewFromData(1, hitdims, NPY_FLOAT64, c2p);
    PyArray_ENABLEFLAGS((PyArrayObject*)c2pArr, NPY_ARRAY_OWNDATA);
    PyDict_SetItemString(maps_dict, "c2p", c2pArr);

    double *s2p = new double[npix];
    for(int i = 0; i < npix; ++i){
        s2p[i] = (*skymap.s2p)[i];
    }
    PyObject *s2pArr = PyArray_SimpleNewFromData(1, hitdims, NPY_FLOAT64, s2p);
    PyArray_ENABLEFLAGS((PyArrayObject*)s2pArr, NPY_ARRAY_OWNDATA);
    PyDict_SetItemString(maps_dict, "s2p", s2pArr);

    double *csp = new double[npix];
    for(int i = 0; i < npix; ++i){
        csp[i] = (*skymap.csp)[i];
    }
    PyObject *cspArr = PyArray_SimpleNewFromData(1, hitdims, NPY_FLOAT64, csp);
    PyArray_ENABLEFLAGS((PyArrayObject*)cspArr, NPY_ARRAY_OWNDATA);
    PyDict_SetItemString(maps_dict, "csp", cspArr);
    return maps_dict;
}

PyObject *PyDriver_wait(PyDriver *self, PyObject* Py_UNUSED(ignored)){
    self->driver->wait();
    Py_RETURN_NONE;
}

std::shared_ptr<dVec> pyobj2sharedvec(PyObject* arr){
    int size = PyArray_Size(arr);
    auto res = std::make_shared<dVec>(size);
    for(int i = 0; i < size; ++i){
        (*res)[i] = *(double*)PyArray_GETPTR1(arr, i);
    }
    return res;
}

std::shared_ptr<VWrapper> pyobj2sharedmap(PyObject *arr){
    int size = PyArray_Size(arr);
    auto res = std::make_shared<VWrapper>((double*)PyArray_GETPTR1(arr, 0), size);
    return res;
}

PyObject *PyDriver_addBeamsys(PyDriver *self, PyObject *args){
    PyObject *params, *maps;
    int ok = PyArg_ParseTuple(args, "OO", &params, &maps);
    if(!ok) return nullptr;

    std::map<int, BeamParams> input;

    Py_ssize_t pos = 0;
    PyObject *key, *val;
    while(PyDict_Next(params, &pos, &key, &val)){
        int freq = PyLong_AsLong(key);
        Py_ssize_t ipos = 0;
        PyObject *ikey, *ival;
        while(PyDict_Next(val, &ipos, &ikey, &ival)){
            if(PyUnicode_CompareWithASCIIString(ikey, "dg") == 0)
                input[freq].dg = pyobj2sharedvec(ival);
            else if(PyUnicode_CompareWithASCIIString(ikey, "dx") == 0)
                input[freq].dx = pyobj2sharedvec(ival);
            else if(PyUnicode_CompareWithASCIIString(ikey, "dy") == 0)
                input[freq].dy = pyobj2sharedvec(ival);
            else if(PyUnicode_CompareWithASCIIString(ikey, "ds") == 0)
                input[freq].ds = pyobj2sharedvec(ival);
            else if(PyUnicode_CompareWithASCIIString(ikey, "dp") == 0)
                input[freq].dp = pyobj2sharedvec(ival);
            else if(PyUnicode_CompareWithASCIIString(ikey, "dc") == 0)
                input[freq].dc = pyobj2sharedvec(ival);
        }
    }

    std::map<int, DerivMaps> derivmaps;
    pos = 0;
    while(PyDict_Next(maps, &pos, &key, &val)){
        int freq = PyLong_AsLong(key);
        Py_ssize_t ipos = 0;
        PyObject *ikey, *ival;
        while(PyDict_Next(val, &ipos, &ikey, &ival)){
            if(PyUnicode_CompareWithASCIIString(ikey, "T") == 0)
                derivmaps[freq].T = pyobj2sharedmap(ival);
            else if(PyUnicode_CompareWithASCIIString(ikey, "T_t") == 0)
                derivmaps[freq].T_t = pyobj2sharedmap(ival);
            else if(PyUnicode_CompareWithASCIIString(ikey, "T_p") == 0)
                derivmaps[freq].T_p = pyobj2sharedmap(ival);
            else if(PyUnicode_CompareWithASCIIString(ikey, "T_tt") == 0)
                derivmaps[freq].T_tt = pyobj2sharedmap(ival);
            else if(PyUnicode_CompareWithASCIIString(ikey, "T_tp") == 0)
                derivmaps[freq].T_tp = pyobj2sharedmap(ival);
            else if(PyUnicode_CompareWithASCIIString(ikey, "T_pp") == 0)
                derivmaps[freq].T_pp = pyobj2sharedmap(ival);
        }
    }
    self->driver->addBeamSysParams(input, derivmaps);
    Py_RETURN_NONE;
}

PyObject *PyDriver_addFittingScansets(PyDriver *self, PyObject *args){
    PyObject *arr;
    PyArg_ParseTuple(args, "O", &arr);
    int size = PyArray_Size(arr);
    auto res = std::make_shared<iVec>(size);
    for(int i = 0; i < size; ++i){
        res->at(i) = *(npy_intp *)PyArray_GETPTR1(arr, i);
    }
    self->driver->addFittingScansets(res);
    Py_RETURN_NONE;
}

PyObject *PyDriver_addFittingTemplate(PyDriver *self, PyObject *args){
    PyObject *maps;
    int ok = PyArg_ParseTuple(args, "O", &maps);
    if(!ok) return nullptr;


    Py_ssize_t pos = 0;
    PyObject *key, *val;

    std::map<int, DerivMaps> derivmaps;
    while(PyDict_Next(maps, &pos, &key, &val)){
        int freq = PyLong_AsLong(key);
        Py_ssize_t ipos = 0;
        PyObject *ikey, *ival;
        while(PyDict_Next(val, &ipos, &ikey, &ival)){
            if(PyUnicode_CompareWithASCIIString(ikey, "T") == 0)
                derivmaps[freq].T = pyobj2sharedmap(ival);
            else if(PyUnicode_CompareWithASCIIString(ikey, "T_t") == 0)
                derivmaps[freq].T_t = pyobj2sharedmap(ival);
            else if(PyUnicode_CompareWithASCIIString(ikey, "T_p") == 0)
                derivmaps[freq].T_p = pyobj2sharedmap(ival);
            else if(PyUnicode_CompareWithASCIIString(ikey, "T_tt") == 0)
                derivmaps[freq].T_tt = pyobj2sharedmap(ival);
            else if(PyUnicode_CompareWithASCIIString(ikey, "T_tp") == 0)
                derivmaps[freq].T_tp = pyobj2sharedmap(ival);
            else if(PyUnicode_CompareWithASCIIString(ikey, "T_pp") == 0)
                derivmaps[freq].T_pp = pyobj2sharedmap(ival);
        }
    }
    self->driver->addFittingTemplate(derivmaps);
    Py_RETURN_NONE;
}

PyObject *beam2npyarr(BeamParams in){
    int ndet = in.dg->size();
    double *data = new double[6*ndet];
    for(int i = 0; i < ndet; ++i){
        data[i] = in.dg->at(i);
        data[i+ndet] = in.dx->at(i);
        data[i+ndet*2] = in.dy->at(i);
        data[i+ndet*3] = in.ds->at(i);
        data[i+ndet*4] = in.dp->at(i);
        data[i+ndet*5] = in.dc->at(i);
    }
    npy_intp dims[2]{6, ndet};
    PyObject *arr = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, data);
    PyArray_ENABLEFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA);
    return arr;
}

PyObject *PyDriver_getFittedParams(PyDriver *self, PyObject* Py_UNUSED(ignored)){
    auto params = self->driver->getFittedParams();
    PyObject *dict = PyDict_New();
    for(auto kv : params){
        PyObject *list = PyList_New(0);
        for(int i = 0; i < kv.second.size(); ++i){
            PyList_Append(list, beam2npyarr(kv.second[i]));
        }
        PyObject *iband = PyLong_FromLong(kv.first);
        PyDict_SetItem(dict, iband, list);
    }
    
    return dict;
}

PyObject *PyDriver_addFittedParams(PyDriver *self, PyObject *args){
    PyObject *dict, *arr;
    PyArg_ParseTuple(args, "OO", &arr, &dict);

    std::map<int, std::vector<BeamParams>> in;
    PyObject *key, *val;
    Py_ssize_t pos = 0;
    while(PyDict_Next(dict, &pos, &key, &val)){
        int iband = PyLong_AsLong(key);
        int size = PyList_GET_SIZE(val);
        in[iband] = std::vector<BeamParams>(size);
        for(int i = 0; i < size; ++i){
            PyObject *arr = PyList_GET_ITEM(val, 0);
            int ndet = PyArray_DIMS(arr)[1];
            in[iband][i] = BeamParams(ndet);
            for(int idet = 0; idet < ndet; ++idet){
                (*in[iband][i].dg)[idet] = *(double*)PyArray_GETPTR2(arr, 0, idet);
                (*in[iband][i].dx)[idet] = *(double*)PyArray_GETPTR2(arr, 1, idet);
                (*in[iband][i].dy)[idet] = *(double*)PyArray_GETPTR2(arr, 2, idet);
                (*in[iband][i].ds)[idet] = *(double*)PyArray_GETPTR2(arr, 3, idet);
                (*in[iband][i].dp)[idet] = *(double*)PyArray_GETPTR2(arr, 4, idet);
                (*in[iband][i].dc)[idet] = *(double*)PyArray_GETPTR2(arr, 5, idet);
            }
        }
    }

    int size = PyArray_Size(arr);
    auto res = std::make_shared<iVec>(size);
    for(int i = 0; i < size; ++i){
        res->at(i) = *(npy_intp *)PyArray_GETPTR1(arr, i);
    }
    // check consistence
    int fitidxSize = res->size();
    for(auto kv : in){
        if (fitidxSize + 1 != kv.second.size()){
            PyErr_Format(PyExc_RuntimeError, "in-consistent!");
            return nullptr;
        }
    }
    self->driver->addFittingScansets(res);
    self->driver->addFittedParams(in);
    Py_RETURN_NONE;
}

PyMethodDef PyDriver_methods[] = {
    {"addScan", (PyCFunction) PyDriver_addScan, METH_VARARGS, "....." },
    {"getMaps", (PyCFunction) PyDriver_getMaps, METH_VARARGS, "....." },
    {"wait", (PyCFunction) PyDriver_wait, METH_VARARGS, "....." },
    {"addBeamsys", (PyCFunction) PyDriver_addBeamsys, METH_VARARGS, "....." },
    {"addFittingTemplate", (PyCFunction) PyDriver_addFittingTemplate, METH_VARARGS, "....." },
    {"addFittingScansets", (PyCFunction) PyDriver_addFittingScansets, METH_VARARGS, "....." },
    {"getFittedParams", (PyCFunction) PyDriver_getFittedParams, METH_VARARGS, "....." },
    {"addFittedParams", (PyCFunction) PyDriver_addFittedParams, METH_VARARGS, "....." },
    {nullptr}  /* Sentinel */
};

PyTypeObject PyDriverType = {
    PyVarObject_HEAD_INIT(NULL, 0)
     "Driver", //.tp_name =

     sizeof(PyDriver), //.tp_basicsize =
     0, //.tp_itemsize =

     (destructor) PyDriver_dealloc, //.tp_dealloc =
     0, //.tp_print =
     0, //.tp_getattr =
     0, //.tp_setattr =
     0, //.tp_as_async=

     0, //.tp_repr =

     0, //.tp_as_number =
     0, //.tp_as_sequence =
     0, //.tp_as_mapping =

     0, //.tp_hash =
     0, //.tp_call =
     0, //.tp_str  =
     0, //.tp_getattro =
     0, //.tp_setattro =
     0, //.tp_as_buffer =
     Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, //.tp_flags =
     "Driver objects", //.tp_doc =
     0, //.tp_traverse =
     0, //.tp_clear =
     0, //.tp_richcompare =
     0, //.tp_weaklistoffset =
     0, //.tp_iter =
     0, //.tp_iternext =
     PyDriver_methods, //.tp_methods =
     0, //.tp_members =
     0, //.tp_getset =
     0, //.tp_base =
     0, //.tp_dict =
     0, //.tp_descr_get =
     0, //.tp_descr_set =
     0, //.tp_dictoffset =
     (initproc) PyDriver_init, //.tp_init =
     0, //.tp_alloc =
     PyDriver_new, //.tp_new =
};

PyMethodDef MyMethods[]={
    {nullptr,nullptr,0,nullptr}
};

PyModuleDef mymod = {
    PyModuleDef_HEAD_INIT,
    "helpers",
    nullptr,
    -1,
    MyMethods
};

PyMODINIT_FUNC PyInit_helper(void)
{
    import_array();
    PyObject *mymodule = PyModule_Create(&mymod);
    if(mymodule == nullptr) return nullptr;
    if(PyType_Ready(&PyDriverType) < 0) return nullptr;
    Py_INCREF(&PyDriverType);
    if (PyModule_AddObject(mymodule, "Driver", reinterpret_cast<PyObject*>(&PyDriverType)) < 0) {
        Py_DECREF(&PyDriverType);
        Py_DECREF(mymodule);
        return nullptr;
    }

    return mymodule;
}
