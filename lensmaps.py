import numpy as np, argparse
from enlib import enmap, powspec, lensing, utils
parser = argparse.ArgumentParser()
parser.add_argument("scal_cls")
parser.add_argument("tens_cls")
parser.add_argument("odir")
parser.add_argument("-s", "--seed", type=int,   default=0)
parser.add_argument("-r", "--res",  type=float, default=0.5)
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-o", "--oversample", type=int, default=1)
args = parser.parse_args()

os    = args.oversample
shape = (3,1024*os,1024*os)
w     = shape[-1]*args.res*np.pi/180/60/os
box   = np.array([[-w,-w],[w,w]])/2
wcs   = enmap.create_wcs(shape, box)

cl_scal, cl_phi = powspec.read_camb_scalar(args.scal_cls, expand=True)
cl_tens = powspec.read_spectrum(args.tens_cls, expand="diag")

## Generate scalar-only lensed and unlensed map
#m_scal_u, m_scal_l = lensing.rand_map(shape, wcs, cl_scal, cl_phi, seed=args.seed, output="ul", verbose=args.verbose)
## Generate tensor-only lensed and unlensed map
#m_tens_u, m_tens_l = lensing.rand_map(shape, wcs, cl_tens, cl_phi, seed=args.seed, output="ul", verbose=args.verbose)

np.random.seed(args.seed)
phi      = enmap.rand_map(shape[-2:], wcs, cl_phi)
m_scal_u = enmap.rand_map(shape, wcs, cl_scal)
m_tens_u = enmap.rand_map(shape, wcs, cl_tens)
m_scal_l = lensing.lens_map_flat(m_scal_u, phi)
m_tens_l = lensing.lens_map_flat(m_tens_u, phi)

# And the sums
m_tot_u = m_scal_u + m_tens_u
m_tot_l = m_scal_l + m_tens_l

# Convert from TQU to TEB and downgrade
def to_eb(m): return enmap.ifft(enmap.map2harm(m)).real
m_scal_u, m_scal_l, m_tens_u, m_tens_l, m_tot_u, m_tot_l = [enmap.downgrade(to_eb(i),os) for i in [m_scal_u, m_scal_l, m_tens_u, m_tens_l, m_tot_u, m_tot_l]]

# And output
utils.mkdir(args.odir)
enmap.write_map(args.odir + "/map_scalar.fits", m_scal_u)
enmap.write_map(args.odir + "/map_tensor.fits", m_tens_u)
enmap.write_map(args.odir + "/map_tot.fits", m_tot_u)
enmap.write_map(args.odir + "/map_scalar_lensed.fits", m_scal_l)
enmap.write_map(args.odir + "/map_tensor_lensed.fits", m_tens_l)
enmap.write_map(args.odir + "/map_tot_lensed.fits", m_tot_l)

