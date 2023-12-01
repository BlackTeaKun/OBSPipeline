import numpy as np
import os

def splitarr(arr, rank, size):
    assert(arr.size % size == 0)
    n_per_rank = arr.size // size
    idx = np.arange(rank * n_per_rank, (rank+1) * n_per_rank)
    return arr[idx].copy()

def splitarrByIdx(arr, idx):
    return arr[idx].copy()

def splitdict(d, rank, size):
    if d is None:
        return None
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            newv = splitdict(v, rank, size)
        elif isinstance(v, np.ndarray):
            newv = splitarr(v, rank, size)
        else:
            raise RuntimeError('??')

        result[k] = newv
    return result

def splitdictByIdx(d, idx):
    if d is None:
        return None
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            newv = splitdictByIdx(v, idx)
        elif isinstance(v, np.ndarray):
            newv = splitarrByIdx(v, idx)
        else:
            raise RuntimeError('??')

        result[k] = newv
    return result


class FocalPlane:
    def __init__(self, path, bands = None):
        self.r     = np.load(os.path.join(path, 'r.npy'))
        self.theta = np.load(os.path.join(path, 'theta.npy'))
        self.chi   = np.load(os.path.join(path, 'chi.npy'))

        self.bands = [95, 150] if bands is None else bands

        self.beamParams = None
        self.groupIdx = None
        self.nblock = None

    
    def addBeamsys(self, path):
        channel = self.bands
        _dblang = 2*(self.theta+self.chi)
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
        self.beamParams = params

    def addCrosstalk(self, path):
        self.groupIdx = np.loadtxt(os.path.join(path, 'readoutGroup.txt'), dtype=int)
        self.nGroup   = np.shape(self.groupIdx)[0]
        self.mixingMatrices = []
        for i in range(self.nGroup):
            self.mixingMatrices.append(
                np.loadtxt(os.path.join(path, 'mixingMatrix', 'group%02d'%i))
            )
        self.mixingMatrices = np.array(self.mixingMatrices)

    def split(self):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        if self.groupIdx is None:
            self.r          = splitarr(self.r, rank, size)
            self.theta      = splitarr(self.theta, rank, size)
            self.chi        = splitarr(self.chi, rank, size)
            self.beamParams = splitdict(self.beamParams, rank, size)
        else:
            blockIdx = np.arange(self.nGroup)
            blockIdx = splitarr(blockIdx, rank, size)

            self.nblock = np.shape(blockIdx)[0]
            idx = self.groupIdx[blockIdx].flatten()

            self.r          = splitarrByIdx(self.r, idx)
            self.theta      = splitarrByIdx(self.theta, idx)
            self.chi        = splitarrByIdx(self.chi, idx)
            self.beamParams = splitdictByIdx(self.beamParams, idx)
            self.mixingMatrices = self.mixingMatrices[blockIdx]

            # preprocessing up-down to sum-diff
            ndet = np.shape(self.mixingMatrices)[1]
            assert(ndet % 2 == 0)
            Pmat = np.zeros((ndet, ndet)) #permutation matrix
            Pmat[np.arange(ndet//2), np.arange(0, ndet, 2)] = 1
            Pmat[np.arange(ndet//2)+ndet//2, np.arange(0, ndet, 2)+1] = 1

            I = np.eye(ndet//2)
            Tmat = np.block([[I, I], [I, -I]])
            transMat = Pmat@Tmat

            self.mixingMatrices = 0.5*np.einsum('il,pij,jk->pkl', transMat, self.mixingMatrices, transMat, optimize='greedy')
            self.mixingMatrices = np.ascontiguousarray(self.mixingMatrices)
        