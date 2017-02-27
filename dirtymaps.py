import numpy as np, argparse, bunch
from enlib import enmap, powspec, lensing, utils
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
parser.add_argument("--sz",     default="-4e2:-10:0.5:-1.75:0:2.5")
parser.add_argument("--glitch", default="1e5:100:0.05:-1.5:1:0.5")
parser.add_argument("--faraday", type=float, default=1e-3)
parser.add_argument("--dust",   default="1e3:1e0:-2.5")
parser.add_argument("--atm",    default="1e9:1e1:-4")
parser.add_argument("--white",  type=float, default=3.7)
parser.add_argument("--angerr", type=float, default=1)
parser.add_argument("--beam",   type=float, default=30)
parser.add_argument("--sidelobe", default="200:10:15:2")
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
	amps = np.maximum(-100*np.abs(info.amp),np.minimum(100*np.abs(info.amp), amps))
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
	print np.max(pmap), np.min(pmap)
	return pmap

def parse_sidelobe(infostr):
	toks = [float(w) for w in infostr.split(":")]
	return bunch.Bunch(amps=np.array([toks[0],toks[1],toks[1]]),width=toks[2]*np.pi/180/60,wave=toks[3])

def sim_sidelobe(shape, wcs, info):
	# Horizontal band for illustration
	pos = enmap.posmap(shape, wcs)
	x  = pos[0]
	x0 = x.reshape(-1)[np.random.randint(x.size)]
	r  = (x-x0)/info.width
	return np.exp(-0.5*r**2)*np.cos(2*np.pi*r/info.wave+np.random.uniform(0,2*np.pi,3)[:,None,None])*info.amps[:,None,None]

def blur(m, sigma):
	l  = np.sum(m.lmap()**2,0)**0.5
	fm = enmap.map2harm(m)
	fm *= np.exp(-0.5*l**2*sigma**2)[None,:,:]
	return enmap.harm2map(fm)

def parse_powlaw(infostr):
	tamp, pamp, alpha = [float(w) for w in infostr.split(":")]
	l = np.arange(10000)
	s = (l+1)**alpha; s[0] = 0
	res = np.zeros((3,3,len(l)))
	res[0,0] = tamp*s
	res[1,1], res[2,2] = pamp*s, pamp*s
	return res

def add(res, m, desc):
	m2 = enmap.ifft(enmap.map2harm(m)).real
	res.tqu.append(m.copy())
	res.teb.append(m2)
	res.desc.append(desc)
	print desc

def output(res, dir):
	utils.mkdir(dir)
	for i, (tqu, teb, desc) in enumerate(zip(res.tqu, res.teb, res.desc)):
		enmap.write_map("%s/%02d_%s_tqu.hdf" % (dir,i+1,desc), tqu)
		enmap.write_map("%s/%02d_%s_teb.hdf" % (dir,i+1,desc), teb)

cl_scal, cl_phi = powspec.read_camb_scalar(args.scal_cls, expand=True)
cl_tens = powspec.read_spectrum(args.tens_cls, expand="diag")
cl_cmb  = cl_scal+cl_tens
np.random.seed(args.seed)

res = bunch.Bunch(tqu=[],teb=[],desc=[])

# 1. Start with last scattering
cmb   = enmap.rand_map(shape, wcs, cl_cmb)
m     = cmb; add(res, m, "cmb")
# 2. Add lensing
phi   = enmap.rand_map(shape[-2:], wcs, cl_phi)
lcmb  = lensing.lens_map_flat(cmb, phi)
m     = lcmb; add(res, m, "lens")
# 3. Add point sources
pinfo = parse_points(args.ptsrc)
pmap  = sim_points(shape, wcs, pinfo)
m    += pmap; add(res, m, "ptsrc")
# 4. Add sz clusters
zinfo = parse_points(args.sz)
zmap  = sim_points(shape, wcs, zinfo)
m    += zmap; add(res, m, "sz")
# 5. Add faraday rotation (probably invisible)
m = enmap.rotate_pol(m, args.faraday)
add(res, m, "faraday")
# 6. Add dust (as a representative galactic foreground)
cl_dust = parse_powlaw(args.dust)
dust = enmap.rand_map(shape, wcs, cl_dust)
m += dust; add(res, m, "dust")
# 7. Add zodiacal light (invisible)
# 8. Add refraction (shift everything by about 1 arcmin for simplicity)
# 9. Add the atmosphere (large-scale blobs, high amplitude)
cl_atm = parse_powlaw(args.atm)
atm = enmap.rand_map(shape, wcs, cl_atm)
m += atm; add(res, m, "atm")
# 10. Add polarization angle errors (3 degrees, say)
m = enmap.rotate_pol(m, args.angerr*np.pi/180)
add(res, m, "polerr")

# 11. Sidelobe
linfo = parse_sidelobe(args.sidelobe)
lobe  = sim_sidelobe(shape, wcs, linfo)
m += lobe; add(res, m, "sidelobe")

# 12. Blur
m = blur(m, args.beam*np.pi/180/60/(8*np.log(2))*0.5)
add(res, m, "beam")

# 13. Add white noise
white = enmap.rand_gauss(shape, wcs)*(np.array([args.white]+[args.white*2**0.5]*2)/args.res)[:,None,None]
m += white; add(res, m, "noise")

# 14. Add glitches (localized features with even higher amplitude)
ginfo = parse_points(args.glitch)
glitch = sim_points(shape, wcs, ginfo)
m += glitch; add(res, m, "glitch")

output(res, args.odir)
