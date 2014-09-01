import numpy as np, argparse, bunch
from enlib import enmap, powspec, lensing, utils
from matplotlib.pylab import *
parser = argparse.ArgumentParser()
parser.add_argument("scal_cls")
parser.add_argument("tens_cls")
parser.add_argument("odir")
parser.add_argument("-s", "--seed", type=int,   default=0)
parser.add_argument("-r", "--res",  type=float, default=0.5)
parser.add_argument("-v", "--verbose", action="store_true")
#parser.add_argument("--ptsrc-power", default="1e-5:1e-8")
#parser.add_argument("--sz-power",    default="0:0")
# Point source distribution: amp:minamp:density:spectral index:polfrac:size
parser.add_argument("--ptsrc",  default="1e3:1:0.5:-1.75:0.02:1.2")
parser.add_argument("--sz",     default="-5e2:-10:0.1:-1.75:0:2.5")
parser.add_argument("--glitch", default="1e5:100:0.05:-1.5:1:0.5")
parser.add_argument("--faraday", type=float, default=1e-3)
parser.add_argument("--dust",   default="1e3:1e0:-2.5")
parser.add_argument("--atm",    default="1e9:1e1:-4")
parser.add_argument("--white",  type=float, default=5)
parser.add_argument("--angerr", type=float, default=3)
args = parser.parse_args()

shape = (3,1024,1024)
w     = shape[-1]*args.res*np.pi/180/60
box   = np.array([[-w,-w],[w,w]])/2
wcs   = enmap.create_wcs(shape, box)

def parse_points(infostr):
	toks = [float(w) for w in infostr.split(":")]
	return bunch.Bunch(amp=toks[0], minamp=toks[1], density=toks[2], alpha=toks[3], pol=toks[4], rad=toks[5]*np.pi/180/60/(8*np.log(2))**0.5)

def sim_points(shape, wcs, info):
	# Simulate the point amplitudes
	N = enmap.area(shape, wcs) * info.density*(180/np.pi)**2
	n = N*(info.minamp/info.amp)**(info.alpha+1)
	amps = info.minamp*np.random.uniform(0,1,n)**(1/(info.alpha+1))
	# Simulate the polarization
	psi  = np.random.uniform(0,np.pi,n)
	amps = amps[None,:] * np.array([psi*0+1,np.cos(2*psi)*info.pol,np.sin(2*psi)*info.pol])
	# Simulate positions uniformly in pixels
	ipos = np.array([np.random.uniform(0,shape[-2],n),np.random.uniform(0,shape[-1],n)])
	pos  = enmap.pix2sky(wcs, ipos)
	# Draw the points on a canvas using convolution. This requires all points
	# to have integer pixel positions and the same radius.
	rawmap = np.zeros(shape)
	for i in range(shape[0]):
		rawmap[i][tuple(ipos.astype(int))] = amps[i]
	l = np.sum(enmap.lmap(shape,wcs)**2,0)**0.5
	kernel = np.exp(-0.5*l**2*info.rad**2)
	# actually perform the convolution
	pmap = enmap.ifft(enmap.fft(rawmap)*kernel[None]).real
	return pmap

def parse_powlaw(infostr):
	tamp, pamp, alpha = [float(w) for w in infostr.split(":")]
	l = np.arange(10000)
	s = (l+1)**alpha; s[0] = 0
	res = np.zeros((3,3,len(l)))
	res[0,0] = tamp*s
	res[1,1], res[2,2] = pamp*s, pamp*s
	return res

cl_scal, cl_phi = powspec.read_camb_scalar(args.scal_cls, expand=True)
cl_tens = powspec.read_spectrum(args.tens_cls, expand="diag")
cl_cmb  = cl_scal+cl_tens
np.random.seed(args.seed)

# 1. Start with last scattering
cmb   = enmap.rand_map(shape, wcs, cl_cmb)
# 2. Add lensing
phi   = enmap.rand_map(shape[-2:], wcs, cl_phi)
lcmb  = lensing.lens_map_flat(cmb, phi)
# 3. Add point sources
pinfo = parse_points(args.ptsrc)
pmap  = sim_points(shape, wcs, pinfo)
# 4. Add sz clusters
zinfo = parse_points(args.sz)
zmap  = sim_points(shape, wcs, zinfo)
m = lcmb+pmap+zmap
# 5. Add faraday rotation (probably invisible)
m = enmap.rotate_pol(m, args.faraday)
# 6. Add dust (as a representative galactic foreground)
cl_dust = parse_powlaw(args.dust)
dust = enmap.rand_map(shape, wcs, cl_dust)
m += dust
# 7. Add zodiacal light (invisible)
# 8. Add refraction (shift everything by about 1 arcmin for simplicity)
# 9. Add the atmosphere (large-scale blobs, high amplitude)
cl_atm = parse_powlaw(args.atm)
atm = enmap.rand_map(shape, wcs, cl_atm)
m += atm
# 10. Add polarization angle errors (3 degrees, say)
m = enmap.rotate_pol(m, args.angerr*np.pi/180)

# 11. Add white noise (probably invisible after adding the atmosphere)
white = enmap.rand_gauss(shape, wcs)*args.white
m += white

# 11. Add glitches (localized features with even higher amplitude)
ginfo = parse_points(args.glitch)
glitch = sim_points(shape, wcs, ginfo)
m += glitch

m2 = enmap.ifft(enmap.map2harm(m)).real
matshow(m2[0], vmin=-500, vmax=500); colorbar()
matshow(m2[1], vmin=-20,  vmax=20);  colorbar()
matshow(m2[2], vmin=-5,   vmax=5);   colorbar(); show()
