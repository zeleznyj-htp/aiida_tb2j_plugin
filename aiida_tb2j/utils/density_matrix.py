import os
import numpy as np
from sisl import Atom, Geometry, Lattice
from sisl.physics import DensityMatrix
from sisl._core.sparse import _ncol_to_indptr
from sisl.io.siesta._help import _mat_spin_convert, _csr_from_siesta
from sisl.io.siesta.binaries import _add_overlap
import sisl._array as _a

def fromfile(file, dtype=float, count=-1, offset=0, like=None):

    dtype = np.dtype(dtype)
    buffer = file.read(dtype.itemsize * count + offset)
    
    return np.frombuffer(buffer, dtype=dtype, count=count, offset=offset, like=None)

def read_info(file):

    no, spin, nsx, nsy, nsz = fromfile(file, dtype=np.int32, count=5, offset=4).T
    nsc = np.stack([nsx, nsy, nsz])
    ncol = fromfile(file, dtype=np.int32, count=no, offset=8)
    col_list = [fromfile(file, dtype=np.int32, count=n, offset=8) for n in ncol]
    col = np.concatenate(col_list)
    dm_list = [np.concatenate([fromfile(file, dtype=np.float64, count=n, offset=8) for n in ncol]) for i in range(spin)]
    dm = np.array(dm_list).T

    return no, spin, nsc, ncol, col, dm

def read_DM(remote_folder, filename='aiida.DM'):    
   
    full_path = os.path.join(remote_folder.get_remote_path(), filename)
    authinfo = remote_folder.get_authinfo()
    
    with authinfo.get_transport() as transport:
        try:
            with transport.sftp.open(full_path, 'rb') as File:
                no, spin, nsc, ncol, col, dm = read_info(File)
        except AttributeError:
            with open(full_path, 'rb') as File:
                no, spin, nsc, ncol, col, dm = read_info(File)
        except:
            raise RuntimeError("Error while reading the density matrix file.")
            
    xyz = [[x, 0, 0] for x in range(no)]
    lattice = Lattice([no, 1, 1], nsc=nsc)
    geom = Geometry(xyz, Atom(1), lattice=lattice)
      
    DM = DensityMatrix(geom, spin, nnzpr=1, dtype=np.float64, orthogonal=False)
    
    DM._csr.ncol = ncol
    DM._csr.ptr = _ncol_to_indptr(ncol)
    DM._csr.col = col - 1
    DM._csr._nnz = len(col)
    
    nnz = np.sum(ncol)
    DM._csr._D = _a.emptyd([nnz, spin+1])
    DM._csr._D[:, :spin] = dm[:, :]
    DM._csr._D[:, spin] = 0.0

    _mat_spin_convert(DM)
    if nsc[0] != 0 or geom.no_s >= col.max():
        _csr_from_siesta(geom, DM._csr)

    DM = DM.transpose(spin=False, sort=True)
    _add_overlap(DM, None, 'dmSileSiesta.read_density_matrix')

    return DM
