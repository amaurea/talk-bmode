import numpy as np, argparse
from enlib import enmap, powspec, gibbs
from scipy import ndimage
parser = argparse.ArgumentParser()
parser.add_argument("template")
parser.add_argument("powspec")
parser.add_argument("ofile")
parser.add_argument("-s", "--sigma", type=float, default=50)
parser.add_argument("-n", "--nsamp", type=int, default=4)
args = parser.parse_args()

def sim_scan_fast(imap, sigma, rad, n):
	mask = np.zeros(imap.shape[-2:])
	mask[rad:-rad,rad:-rad] = 1
	w = np.maximum(1 - ndimage.distance_transform_edt(1-mask)/rad,0)**3
	w = w[None,:,:]*np.array([1,0.5,0.5])[:,None,None]*sigma**-2*n
	w = enmap.samewcs(w, imap)
	m = imap + np.random.standard_normal(imap.shape)*w**-0.5
	m[~np.isfinite(m)] = 0
	return m, w

template = enmap.read_map(args.template)[:,::2,::2]
ps       = powspec.read_spectrum(args.powspec, expand="diag")
sim      = enmap.rand_map(template.shape, template.wcs, ps)

# Simulate noisy data
m, w = sim_scan_fast(sim, args.sigma, 128, 1)
ncomp = m.shape[0]
iN = enmap.zeros((ncomp,ncomp)+w.shape[-2:])
for i in range(ncomp): iN[i,i] = w[i]
iS = enmap.spec2flat(m.shape, m.wcs, ps, -1)

# Draw samples
sampler = gibbs.FieldSampler(iS, iN, m)

res = enmap.zeros((1+args.nsamp,)+m.shape,m.wcs)
res[0] = m
for i in range(args.nsamp):
	res[i] = sampler.sample(True)
enmap.write_map(args.ofile, res)
