import pickle
import numpy as np
import sys
import glob
import os
import healpy as hp

outdir = sys.argv[2]

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def merge(a, b):
    if not isinstance(a, dict):
        return a + b
    items = [*a.items(), *b.items()]
    cross = [(k, merge(a[k], b[k])) for k in a.keys() & b.keys()]
    return dict(items + cross)

def solve(paths):
    if len(paths) == 0:
        return
    maps = {}
    for path in paths:
        print(path)
        data = load(path)
        maps = merge(maps, data) 

    mask = maps['hitmap'] > 4

    result = {}
    for v in [95, 150]:
        tqu = np.zeros((3, mask.size))

        tqu[0, mask] = maps[v]['tsum'][mask]/maps['hitmap'][mask]
        det = maps['c2p']*maps['s2p'] - maps['csp']**2
        tqu[1, mask] = (maps['s2p'] * maps[v]['cpd'] - maps['csp']*maps[v]['spd'])[mask] / det[mask]
        tqu[2, mask] = (-maps['csp'] * maps[v]['cpd'] + maps['c2p']*maps[v]['spd'])[mask] / det[mask]

        result[v] = tqu
    result['hitmap'] = maps['hitmap']

    return result

def save(result, path, prefix):
    if result is None:
        return
    os.makedirs(path, exist_ok=True)
    hp.write_map(os.path.join(path, 'HITMAP.fits'), result['hitmap'], overwrite=True)
    for v in [95, 150]:
        name = f'Ali_{v}.fits'
        if prefix:
            name = f'{prefix}_' + name
        hp.write_map(os.path.join(path, name), result[v])



pattern = f'./{sys.argv[1]}/map_before*'
before = glob.glob(pattern)
result = solve(before)
save(result, outdir, '')

pattern  = f'./{sys.argv[1]}/map_after*'
before = glob.glob(pattern)
result = solve(before)
save(result, outdir, 'DEPROJ')


