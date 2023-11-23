import sys
sys.path.append('./cpp2/')
import helper
import platform
nodeinfo = platform.uname()

import pickle
import numpy as np
import tqdm
import os
import astropy.units as u
print(f'node: {nodeinfo.node} is importing mpi4py')
from mpi4py import MPI
print(f'node: {nodeinfo.node} done')
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
nside = 1024

scandir = './scan/'

outdir = sys.argv[1]

class CES:
    def __init__(self, ra, dec, pa):
        self.ra = ra
        self.dec = dec
        self.pa = pa
        self.nsamp = ra.size

class Scanset:
    def __init__(self, trace, st, ed):
        self.ces = []
        for i in range(len(st)):
            self.ces.append(CES(trace[0, st[i]:ed[i]], trace[1, st[i]:ed[i]], trace[2, st[i]:ed[i]]))
        self.nces = len(self.ces)

class Scan:
    def __init__(self, path):
        self.dict_names = ['ces_st',
                           'ces_ed',
                           'ces_prev_elnod',
                           'ces_succ_elnod',
                           'elnod_st',
                           'elnod_ed',
                           'tod_fname',
                           'mnt_fname',
                           'file_length',
                           'file_cumst',
                           'ces_time_st',
                           'ces_time_ed',
                           'ces_az_st',
                           'ces_az_ed',
                           'ces_daz',
                           'ces_el',
                           'ces_type']
        self.scans = {}
        with open(path, 'rb') as f:
            for name in self.dict_names:
                self.scans[name] = pickle.load(f)

        self.nscansets = 0
        for day in self.scans['ces_st']:
            scanst = self.scans['ces_st'][day]
            self.nscansets = self.nscansets + len(scanst)
    def loadday(self, day):
        file_length = self.scans['file_length'][day]
        file_cumst = self.scans['file_cumst'][day]
        nfile = len(file_length)
        length = np.sum(file_length)
        scan_trace = np.zeros((3, length))
        for i in range(nfile):
            ist = file_cumst[i]
            ied = ist + file_length[i]
            scan_trace[:, ist:ied] = np.load(scandir + day + f'/BS{day}P{i:03}.npy')

        self.trace = scan_trace
        self.scanst = self.scans['ces_st'][day]
        self.scaned = self.scans['ces_ed'][day]


class FP:
    def __init__(self, r, theta, chi):
        self.r = np.ascontiguousarray(r[::4])
        self.theta = np.ascontiguousarray(theta[::4])
        self.chi = np.ascontiguousarray(chi[::4])
        self.ndet = self.r.size

def loadbeamsys(path, theta, chi):
    channel = [95, 150]
    _dblang = 2*(theta+chi)
    params = {}
    for nu in channel:
        dg     = np.loadtxt(os.path.join(path, 'dg_%dG.txt'%(nu))).T[2]
        dsigma = np.loadtxt(os.path.join(path, 'dsigma_%dG.txt'%(nu))).T[2]
        dx_fp  = np.loadtxt(os.path.join(path, 'dx_%dG.txt'%(nu))).T[2]
        dy_fp  = np.loadtxt(os.path.join(path, 'dy_%dG.txt'%(nu))).T[2]
        de_mod = np.loadtxt(os.path.join(path, 'de_conti_%dG.txt'%(nu))).T[2]
        de_ori = np.loadtxt(os.path.join(path, 'de_orient_%dG.txt'%(nu))).T[2]

        # pointing difference
        dx_fp  = np.deg2rad(dx_fp)
        dy_fp  = np.deg2rad(dy_fp)
        dx_lcl = np.cos(_dblang) * dx_fp + np.sin(_dblang) * dy_fp
        dy_lcl = -np.sin(_dblang) * dx_fp + np.cos(_dblang) * dy_fp

        # dsigma to dfwhm

        # eccentricity (temporary solution of jjin)
        # Here, base on the defination of de_ori by jjin, de_ori represents
        # the angle between the major axis of the beam difference map, at
        # linear order the beam difference map can be treated as an indivi-
        # dual ecliptic beam map, and the favour orientation of UP detector.
        dp = de_mod * np.cos(_dblang + 2.*de_ori)
        dc = de_mod * np.sin(_dblang + 2.*de_ori)

        dg = np.ascontiguousarray(dg)
        ds = np.ascontiguousarray(dsigma)
        dx = np.ascontiguousarray(dx_lcl)
        dy = np.ascontiguousarray(dy_lcl)
        dp = np.ascontiguousarray(dp)
        dc = np.ascontiguousarray(dc)

        params[nu] = {}
        params[nu]['dg'] = dg
        params[nu]['dx'] = dx
        params[nu]['dy'] = dy
        params[nu]['ds'] = ds
        params[nu]['dp'] = dp
        params[nu]['dc'] = dc
    return params

def loadmaps(path, bands):
    ms = {}
    derivs = {}
    template = {}
    for v in bands:
        ms[v] = np.load(os.path.join(path, f'A{v}.npy'))
        with open(os.path.join(path, f'A{v}_der'), 'rb') as f:
            derivs[v] = pickle.load(f)
        with open(os.path.join(path, f'P{v}_der'), 'rb') as f:
            template[v] = pickle.load(f)

    return ms, derivs, template

def splitarr(arr, rank, size):
    assert(arr.size % size == 0)
    n_per_rank = arr.size // size
    idx = np.arange(rank * n_per_rank, (rank+1) * n_per_rank)
    return arr[idx].copy()

def splitdict(d, rank, size):
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            newv = splitdict(v, rank, size)
        elif isinstance(v, np.ndarray):
            newv = splitarr(v, rank, size)
        else:
            raise RuntimeError('??')

        result[k] = newv
    return result;


HFIbeam = {
    100: 9.68200 * u.arcmin,
    143: 7.30300 * u.arcmin
}
Alibeam = {
    95: 19 * u.arcmin,
    150: 11 * u.arcmin
}

ali2hfi = {
    95: 100,
    150: 143
}

fp_r = np.load('./fp/r.npy')
fp_theta = np.load('./fp/theta.npy')
fp_chi = np.load('./fp/chi.npy')
beamsys = loadbeamsys('./beamsys/', fp_theta, fp_chi)

fp_r = splitarr(fp_r, rank, size)
fp_theta = splitarr(fp_theta, rank, size)
fp_chi = splitarr(fp_chi, rank, size)
beamsys = splitdict(beamsys, rank, size)

print('loading scan')
s = Scan(scandir+'scancfg_all.pkl')
nscansets = s.nscansets
daylist = list(s.scans['ces_st'].keys())

bands = [95, 150]
ms, derivs, temp = loadmaps(outdir, bands)


beams = {
    95: (19 * u.arcmin).to(u.rad).value,
    150: (11 * u.arcmin).to(u.rad).value
}

def dump(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

fitidx = np.arange(30, nscansets-30, 30)


driver = helper.Driver(nside, ms, beams, fp_r, fp_theta, fp_chi)
driver.addBeamsys(beamsys, derivs)
driver.addFittingTemplate(temp)
driver.addFittingScansets(fitidx)

comm.Barrier()
for day in daylist:
    print('1', day)
    s.loadday(day)
    comm.Barrier()
    cesst = s.scanst
    cesed = s.scaned
    trace = s.trace
    nscansets = len(cesst)
    for i in tqdm.tqdm(range(nscansets)):
    # for i in tqdm.tqdm(range(4)):
        driver.addScan(*trace, cesst[i], cesed[i])

maps = driver.getMaps()
fitted_params = driver.getFittedParams()
# hitmap = maps['hitmap']
# mask = hitmap >= 4
# print(maps[95][:, mask])
# print(ms[95][:, mask])
# exit()

dump(os.path.join(outdir, f'map_before{rank}'), maps)
dump(os.path.join(outdir, f'params{rank}'), fitted_params)


print('prepareing second run')
#second run
driver = helper.Driver(nside, ms, beams, fp_r, fp_theta, fp_chi)
driver.addBeamsys(beamsys, derivs)
driver.addFittedParams(fitidx, fitted_params)
comm.Barrier()
for day in daylist:
    print('2', day)
    s.loadday(day)
    comm.Barrier()
    cesst = s.scanst
    cesed = s.scaned
    trace = s.trace
    nscansets = len(cesst)
    for i in tqdm.tqdm(range(nscansets)):
        driver.addScan(*trace, cesst[i], cesed[i])


maps = driver.getMaps()
dump(os.path.join(outdir, f'map_after{rank}'), maps)

