import numpy as np, argparse
from scipy import ndimage
from enlib import enmap
parser = argparse.ArgumentParser()
parser.add_argument("ifile")
parser.add_argument("ofile")
parser.add_argument("-r", "--apod-radius", type=int, default=64)
args = parser.parse_args()

def make_apod(shape, rad):
	mask = np.zeros(shape[-2:])
	mask[rad:-rad,rad:-rad] = 1
	w = np.maximum(1 - ndimage.distance_transform_edt(1-mask)/rad,0)**3
	return w

teb = enmap.read_map(args.ifile)
tqu = enmap.harm2map(enmap.fft(teb))
mask = make_apod(teb.shape, args.apod_radius)

tqu_mask = tqu*mask[None]
teb_mask = enmap.ifft(enmap.map2harm(tqu_mask)).real

res = enmap.samewcs([teb,tqu,tqu_mask,teb_mask], teb)
enmap.write_map(args.ofile, res)
