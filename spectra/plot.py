import numpy as np, matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt, scale as mscale, transforms as mtransforms, ticker

def read_lxy(fname):
	return np.loadtxt(fname).T
def read_1l2xyy(fname):
	data = np.loadtxt(fname).T
	return np.array([data[1],data[3],np.mean(data[4:6],0)])
def read_l12xyy(fname):
	data = np.loadtxt(fname).T
	return np.array([data[0],data[3],np.mean(data[4:6],0)])
def read_l12xy(fname):
	data = np.loadtxt(fname).T
	return data[[0,3,4]]
def read_l12ttteeebb(fname):
	data = np.loadtxt(fname).T
	return data[[0,3,4]], data[[0,7,8]], data[[0,9,10]]
def read_1l2y_lim(fname):
	return np.loadtxt(fname).T[[0,2,3]]
def read_ltteebb(fname):
	data = np.loadtxt(fname).T
	return data[[0,1,2]], data[[0,3,4]], data[[0,5,6]]
def read_12y_lim(fname):
	return np.loadtxt(fname).T
def read_1l2eebb(fname):
	data = np.loadtxt(fname).T
	return data[[1,3,4]], data[[1,5,6]]
def read_spt(fname, col=0):
	data = np.loadtxt(fname).T
	return data[[2,3+2*col,4+2*col]]
def read_quiet(fname):
	data = np.loadtxt(fname).T
	l = np.mean(data[:2],0)
	vals = data[2:-1:3]
	errs = (data[3:-1:3]-data[4:-1:3])/2
	return tuple([np.array([l,v,e]) for v,e in zip(vals,errs)] + [data[[0,1,-1]]])
def read_bicep2(fname):
	data = np.loadtxt(fname).T
	l = data[1]
	return tuple([np.array([l,v,e]) for v,e in zip(data[3:9],data[9:])])
def read_sptpol_indirect(fname):
	data = np.loadtxt(fname).T
	data[1:] *= data[0]**2/(2*np.pi)
	return data

# planck
planck_tt = read_l12xyy("planck_tt.txt")
planck_ee = read_lxy("planck_ee.txt")
# spt
spt_tt_11 = read_spt("spt_tt_11.txt",3)
spt_tt_14 = read_spt("spt_tt_14.txt",3)
sptpol_bb = read_sptpol_indirect("sptpol_bb_indirect2.txt")
# act
act_tt = read_lxy("act_148_tt.txt")
actpol_tt, actpol_ee, actpol_bb = read_l12ttteeebb("actpol.txt")
_, _, actpol_bb_rebin = read_l12ttteeebb("actpol_rebin.txt")
# wmap
wmap_tt = read_l12xy("wmap_binned_tt_spectrum_9yr_v5.txt")
wmap_ee = read_l12xy("wmap_binned_ee_spectrum_9yr_v5.txt")
wmap_bb = read_l12xy("wmap_binned_bb_spectrum_9yr_v5.txt")
# bicep 1
bicep1_bblim = read_1l2y_lim("bicep1_bblim.txt")
# bicep 2
bicep2_tt, _, bicep2_ee, bicep2_bb, _, _ = read_bicep2("bicep2.txt")
# quad
quad_tt, quad_ee, quad_bb = read_ltteebb("quad_2009.txt")
quad_bblim = read_12y_lim("quad_bblim.txt")
# polarbear
polarbear_ee, polarbear_bb = read_1l2eebb("polarbear_ee_bb.txt")
# quiet
quiet_q_ee, quiet_q_bb, _, quiet_q_bblim = read_quiet("quiet_q.txt")
_, quiet_w_ee, quiet_w_bb, _, quiet_w_bblim = read_quiet("quiet_w.txt")

# Theory curves
theory_scal_lensed = np.loadtxt("bicepfit/bicep2_planck_lensedcls.dat")[:8000].T
theory_tens        = np.loadtxt("bicepfit/bicep2_planck_tenscls.dat")[:8000].T
theory_tens[1:,5000:] = 0 # hack: fix camb glitch
theory_tot = theory_scal_lensed + theory_tens; theory_tot[0] = theory_tens[0]

#def scalefun(l): return l**2/(2*np.pi)
def scalefun(l): return l*0+1

def plot(data, color, label, maxerr=0.5):
	mask = (data[2]/data[1] < maxerr)*(data[1]>0) + (maxerr is None)
	data = data[:,mask]
	scale = scalefun(data[0])
	plt.errorbar(data[0], data[1]/scale, yerr=data[2]/scale, fmt=".", color=color, elinewidth=3, capsize=0, label=label)
def plotlim(data, color, label, maxval=6):
	l, w = (data[1]+data[0])/2, (data[1]-data[0])/2
	scale = scalefun(l)
	mask = data[2]/scale<maxval
	l, w, data, scale = l[mask], w[mask], data[:,mask], scale[mask]
	plt.errorbar(l, data[2]/scale, xerr=w, marker="", color=color, label=label, elinewidth=2, capsize=0)
def plot_theory(x, y, color, label):
	scale = scalefun(x)
	plt.plot(x, y/scale, label=label, color=color, linestyle="-")
def plot_theory_range(data1, data2, color, label):
	scale = scalefun(data1[0])
	plt.fill_between(data1[0], data1[1], data2[1], facecolor=color, alpha=0.25, color=color, linewidth=1)
	plt.plot(data1[0], data1[1]/scale, label=label, color=color, linewidth=0.5, linestyle="-")
	plt.plot(data1[0], data2[1]/scale, label=label, color=color, linewidth=0.5, linestyle="-")

def plot_mainfig(ofile, after):
	plt.rc('lines', linestyle='None')
	plt.figure()

	plot_theory_range(theory_scal_lensed[[0,1]], (theory_tens*0.5+theory_scal_lensed)[[0,1]], "black", None)
	plot_theory_range(theory_scal_lensed[[0,2]], (theory_tens*0.5+theory_scal_lensed)[[0,2]], "black", None)
	plot_theory_range(theory_scal_lensed[[0,3]], (theory_tens*0.5+theory_scal_lensed)[[0,3]], "black", None)
	if after: plot_theory(theory_tot[0], theory_tot[3], "black", None)

	# TT part
	plot(planck_tt, "red", "Planck")
	plot(act_tt, "green", "ACT")
	plot(spt_tt_14 if after else spt_tt_11, "blue", "SPT")
	# EE part
	plot(wmap_ee,   "orange", "WMAP")
	plot(quiet_w_ee, "purple", "QUIET W")
	plot(polarbear_ee, "cyan", "POLARBEAR")
	plot(planck_ee, "red", None)
	if after:
		plot(actpol_ee, "green", None)
		plot(bicep2_ee, "black", None)
	# BB part
	plotlim(bicep1_bblim, "brown", "BICEP 1")
	plotlim(quiet_w_bblim, "purple", None)
	plotlim(quad_bblim, "pink", "QUAD")
	plot(polarbear_bb, "cyan", None, maxerr=None)
	plot(sptpol_bb, "blue", None, maxerr=None)
	if after: plot(bicep2_bb, "black", "BICEP 2", maxerr=None)

	plt.yscale("log")
	plt.xscale("symlog", linthreshx=500)
	plt.xlim([0,8000])
	plt.ylim([1e-3,1e4])
	plt.xticks([100,200,300,400,500,1000,2000,4000,8000])
	plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
	#plt.legend(numpoints=1)
	plt.grid()
	plt.xlabel('$\ell$')
	plt.ylabel('$D_\ell$ ($\mu K^2$)')
	plt.savefig(ofile)

# Zoom on low-l TT
def plot_lowl_tt(ofile):
	plt.figure()
	plot(planck_tt, "red", "Planck")
	plot_theory_range(theory_scal_lensed[[0,1]], (theory_tens*0.5+theory_scal_lensed)[[0,1]], "black", None)
	plt.xlim([0,50])
	plt.ylim([0,2000])
	plt.grid()
	plt.xlabel('$\ell$')
	plt.ylabel('$D_\ell$ ($\mu K^2$)')
	plt.savefig(ofile)

plot_mainfig("spectra_before.pdf", False)
plot_mainfig("spectra_after.pdf", True)

plot_lowl_tt("spectra_lowl_tt.pdf")
