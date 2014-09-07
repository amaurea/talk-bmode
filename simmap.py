import numpy as np, argparse
from scipy import ndimage
from enlib import enmap
from matplotlib.pylab import *
parser = argparse.ArgumentParser()
parser.add_argument("ifile")
parser.add_argument("ofile")
parser.add_argument("-s", "--scan-noise",   type=float, default=500)
parser.add_argument("-t", "--target-noise", type=float, default=1)
args = parser.parse_args()

def sim_scan(imap, sigma, sshape, rad):
	w = enmap.zeros(imap.shape, imap.wcs)+(np.array([1,0.5,0.5])/sigma**2)[:,None,None]
	m = imap + np.random.standard_normal(imap.shape)*w**-0.5
	# Draw random center pos
	r, phi = np.random.uniform(0,rad), np.random.uniform(0,2*np.pi)
	y,x = (np.array(imap.shape[-2:])/2 + np.array([r*np.cos(phi),r*np.sin(phi)])).astype(int)
	H,W = sshape
	mask = np.zeros(imap.shape)+1
	mask[:,:y-H/2,:] = 0
	mask[:,y+H/2:,:] = 0
	mask[:,:,:x-W/2] = 0
	mask[:,:,x+W/2:] = 0
	return m*mask, w*mask

def sim_scan_fast(imap, sigma, rad, n):
	mask = np.zeros(imap.shape[-2:])
	mask[rad:-rad,rad:-rad] = 1
	w = np.maximum(1 - ndimage.distance_transform_edt(1-mask)/rad,0)**3
	w = w[None,:,:]*np.array([1,0.5,0.5])[:,None,None]*sigma**-2*n
	w = enmap.samewcs(w, imap)
	m = imap + np.random.standard_normal(imap.shape)*w**-0.5
	m[np.isnan(m)] = 0
	return m, w
imap = enmap.read_map(args.ifile)

edge_width = 256
scan_shape = [d-2*edge_width for d in imap.shape[-2:]]

mtot, wtot = imap*0, imap*0
n = 0
outmaps = []
for ostep in range(0,int(np.log((args.scan_noise/args.target_noise)**2)/np.log(2))+1):
	nloc = 2**ostep
	if nloc < 512:
		for i in range(0,nloc):
			m,mw = sim_scan(imap, args.scan_noise, scan_shape, edge_width)
			mtot += m*mw
			wtot += mw
	else:
		m,mw = sim_scan_fast(imap, args.scan_noise, edge_width, nloc)
		mtot += m*mw
		wtot += mw
	outmaps.append(mtot/wtot)
	print "%5d %9.3f" % (2**(ostep+1)-1, np.max(wtot)**-0.5)
outmaps = enmap.samewcs(outmaps, imap)
enmap.write_map(args.ofile, outmaps)
