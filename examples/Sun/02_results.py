import numpy as np 
import emcee
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')
import corner
labels = np.array(["eta2", "eta3", "eta4", 
                   "weight1", "weight2", "weight3",
                   "slope1", "offset1", "slope2", "offset2", "slope3", "offset3", 
                   "jitter1", "jitter2", "jitter3", "logl"])

from artgpn.arttwo import network
from artgpn import weight, node, mean

time,rv,rverr,bis,biserr, fw, fwerr = np.loadtxt("sun50points.txt",
                                                 skiprows = 1, unpack = True, 
                                                 usecols = (0,1,2,7,8,9,10))
time, val1, val1err = time, rv, rverr
val2,val2err = bis, biserr
val3,val3err = fw, fwerr

filename =  "savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)

storage_name = "09_results.txt"
f = open(storage_name, "a")

print('iterations:', sampler.iteration, file=f)
#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.5 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
print("burn-in: {0}".format(burnin), file=f)
print("thin: {0}".format(thin), file=f)
print("flat chain shape: {0}".format(samples.shape), file=f)
all_samples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)

labels = np.array(["eta2", "eta3", "eta4", 
                   "weight1", "weight2", "weight3",
                   "slope1", "offset1", "slope2", "offset2", "slope3", "offset3", 
                   "jitter1", "jitter2", "jitter3","logPost"])
corner.corner(all_samples, labels=labels, color="k", bins = 50,
              quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True, 
              show_titles=True, plot_density=True, plot_contours=True,
              fill_contours=True, plot_datapoints=False)
plt.savefig('11_cornerPlot_logP.png')
plt.close('all')

from priors import priors
prior = priors()

logLike = log_prob_samples
for i, j in enumerate(samples):
    n2,n3,n4,w11,w12,w13,s1,off1,s2,off2,s3,off3,j1,j2,j3 = j
    logcdfEta2 = prior[1].logcdf(0.5*n3)
    logprior = (prior[1].logpdf(n2)-logcdfEta2)
    logprior += prior[2].logpdf(n3) 
    logprior += prior[3].logpdf(n4)
    logprior += prior[4].logpdf(w11)
    logprior += prior[6].logpdf(w12)
    logprior += prior[8].logpdf(w13)
    logprior += prior[10].logpdf(s1)
    logprior += prior[11].logpdf(off1)
    logprior += prior[12].logpdf(s2)
    logprior += prior[13].logpdf(off2)
    logprior += prior[14].logpdf(s3)
    logprior += prior[15].logpdf(off3)
    logprior += prior[16].logpdf(j1)
    logprior += prior[17].logpdf(j2)
    logprior += prior[18].logpdf(j3)
    logLike[i] = logLike[i] - logprior
all_samples = np.concatenate((samples, logLike[:, None]), axis=1)
labels = np.array(["eta2", "eta3", "eta4", 
                   "weight1", "weight2", "weight3",
                   "slope1", "offset1", "slope2", "offset2", "slope3", "offset3", 
                   "jitter1", "jitter2", "jitter3", "logLike"])
corner.corner(all_samples, labels=labels, color="k", bins = 50,
              quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True, 
              show_titles=True, plot_density=True, plot_contours=True,
              fill_contours=True, plot_datapoints=True)
plt.savefig('12_cornerPlot_logL.png')
plt.close('all')

plt.rcParams['figure.figsize'] = [15, 5]
n2, n3,n4,w11,w12,w13,s1,off1,s2,off2,s3,off3,j1,j2,j3,logl = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(all_samples, [16, 50, 84],axis=0)))
print('Medians', file=f)
print(file=f)
print('eta 2= {0[0]} +{0[1]} -{0[2]}'.format(n2), file=f)
print('eta 3= {0[0]} +{0[1]} -{0[2]}'.format(n3), file=f)
print('eta 4= {0[0]} +{0[1]} -{0[2]}'.format(n4), file=f)
print(file=f)
print('weight 1= {0[0]} +{0[1]} -{0[2]}'.format(w11), file=f)
print('weight 2= {0[0]} +{0[1]} -{0[2]}'.format(w12), file=f)
print('weight 3= {0[0]} +{0[1]} -{0[2]}'.format(w13), file=f)
print(file=f)
print('RV slope = {0[0]} +{0[1]} -{0[2]}'.format(s1), file=f)
print('RV offset = {0[0]} +{0[1]} -{0[2]}'.format(off1), file=f)
print('BIS slope = {0[0]} +{0[1]} -{0[2]}'.format(s2), file=f)
print('BIS offset = {0[0]} +{0[1]} -{0[2]}'.format(off2), file=f)
print('RV offset = {0[0]} +{0[1]} -{0[2]}'.format(off1), file=f)
print('FWHM slope = {0[0]} +{0[1]} -{0[2]}'.format(s3), file=f)
print('FWHM offset = {0[0]} +{0[1]} -{0[2]}'.format(off3), file=f)
print(file=f)
print('RVs jitter = {0[0]} +{0[1]} -{0[2]}'.format(j1), file=f)
print('BIS jitter = {0[0]} +{0[1]} -{0[2]}'.format(j2), file=f)
print('FWHM jitter = {0[0]} +{0[1]} -{0[2]}'.format(j3), file=f)
print('loglike= {0[0]} +{0[1]} -{0[2]}'.format(logl), file=f)
print(file=f)

#Having a solution from our MCMC we need to redefine all the network
nodes = [node.QuasiPeriodic(n2[0], n3[0], n4[0])]
weights = [weight.Constant(w11[0]), 
           weight.Constant(w12[0]),
           weight.Constant(w13[0])]
means = [mean.Linear(s1[0], off1[0]),
         mean.Linear(s2[0], off2[0]),
         mean.Linear(s3[0], off3[0])]
jitters = [j1[0], j2[0], j3[0]]

GPnet = network(1, time, val1, val1err, val2, val2err, val3, val3err)

#And then plot our fit to see if looks good
extention = 5
mu11, std11, cov11 = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                                      jitters = jitters,
                                      time = np.linspace(time.min()-extention, time.max()+extention, 5000),
                                      dataset = 1)
mu22, std22, cov22 = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                                      jitters = jitters,
                                      time = np.linspace(time.min()-extention, time.max()+extention, 5000),
                                      dataset = 2)
mu33, std33, cov33 = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                                      jitters = jitters,
                                      time = np.linspace(time.min()-extention, time.max()+extention, 5000),
                                      dataset = 3)
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3,ncols=1, sharex=True)
ax1.set_title('Fits')
ax1.fill_between(np.linspace(time.min()-extention, time.max()+extention, 5000), 
                 mu11+std11, mu11-std11, color="red", alpha=0.25)
ax1.plot(np.linspace(time.min()-extention, time.max()+extention, 5000), mu11, "r-", alpha=1, lw=1.5)
ax1.errorbar(time,val1, val1err, fmt = "k.")
ax1.set_ylabel("RVs")
ax2.fill_between(np.linspace(time.min()-extention, time.max()+extention, 5000), 
                 mu22+std22, mu22-std22, color="red", alpha=0.25)
ax2.plot(np.linspace(time.min()-extention, time.max()+extention, 5000), mu22, "r-", alpha=1, lw=1.5)
ax2.errorbar(time,val2, val2err, fmt = "k.")
ax2.set_ylabel("BIS")
ax3.fill_between(np.linspace(time.min()-extention, time.max()+extention, 5000), 
                 mu33+std33, mu33-std33, color="red", alpha=0.25)
ax3.plot(np.linspace(time.min()-extention, time.max()+extention, 5000), mu33, "r-", alpha=1, lw=1.5)
ax3.errorbar(time,val3, val3err, fmt = "k.")
ax3.set_ylabel("FWHM")
plt.savefig('13_mediansPlot.png')
plt.close('all')

#checking the likelihood that matters to us
values = np.where(all_samples[:,-1] == np.max(all_samples[:,-1]))
opt_samples = all_samples[values,:]
opt_samples = opt_samples.reshape(-1, 16)
#Having a solution from our MCMC we need to redefine all the network
nodes = [node.QuasiPeriodic(opt_samples[-1,0], opt_samples[-1,1], opt_samples[-1,2])]
weights = [weight.Constant(opt_samples[-1,3]), 
           weight.Constant(opt_samples[-1,4]),
           weight.Constant(opt_samples[-1,5])]
means = [mean.Linear(opt_samples[-1,6], opt_samples[-1,7]),
         mean.Linear(opt_samples[-1,8], opt_samples[-1,9]),
         mean.Linear(opt_samples[-1,10], opt_samples[-1,11])]
jitters = [opt_samples[-1,12], opt_samples[-1,13], opt_samples[-1,14]]

GPnet = network(1, time, val1, val1err, val2, val2err,  val3, val3err)

#And then plot our fit to see if looks good
extention = 5
tstar = np.linspace(time.min()-extention, time.max()+extention, 5000)
mu11, std11, cov11 = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                                      jitters = jitters,time = tstar, dataset = 1)
mu22, std22, cov22 = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                                      jitters = jitters, time = tstar, dataset = 2)
mu33, std33, cov33 = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                                      jitters = jitters, time = tstar, dataset = 3)
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3,ncols=1, sharex=True)
ax1.set_title('Fits')
ax1.fill_between(np.linspace(time.min()-extention, time.max()+extention, 5000), 
                 mu11+std11, mu11-std11, color="red", alpha=0.25)
ax1.plot(np.linspace(time.min()-extention, time.max()+extention, 5000), mu11, "r-", alpha=1, lw=1.5)
ax1.errorbar(time,val1, val1err, fmt = "k.")
ax1.set_ylabel("RVs")
ax2.fill_between(np.linspace(time.min()-extention, time.max()+extention, 5000), 
                 mu22+std22, mu22-std22, color="red", alpha=0.25)
ax2.plot(np.linspace(time.min()-extention, time.max()+extention, 5000), mu22, "r-", alpha=1, lw=1.5)
ax2.errorbar(time,val2, val2err, fmt = "k.")
ax2.set_ylabel("BIS")
ax3.fill_between(np.linspace(time.min()-extention, time.max()+extention, 5000), 
                 mu33+std33, mu33-std33, color="red", alpha=0.25)
ax3.plot(np.linspace(time.min()-extention, time.max()+extention, 5000), mu33, "r-", alpha=1, lw=1.5)
ax3.errorbar(time,val3, val3err, fmt = "k.")
ax3.set_ylabel("FWHM")
plt.savefig('14_mapPlot.png')
plt.close('all')

print('MAP values', file=f)
print(file=f)
print('eta 2= {0}'.format(opt_samples[-1,0]), file=f)
print('eta 3= {0}'.format(opt_samples[-1,1]), file=f)
print('eta 4= {0}'.format(opt_samples[-1,2]), file=f)
print(file=f)
print('weight 1= {0}'.format(opt_samples[-1,3]), file=f)
print('weight 2= {0}'.format(opt_samples[-1,4]), file=f)
print('weight 3= {0}'.format(opt_samples[-1,5]), file=f)
print(file=f)
print('RV slope = {0}'.format(opt_samples[-1,6]), file=f)
print('RV offset = {0}'.format(opt_samples[-1,7]), file=f)
print('BIS slope = {0}'.format(opt_samples[-1,8]), file=f)
print('BIS offset = {0}'.format(opt_samples[-1,9]), file=f)
print('FWHM slope = {0}'.format(opt_samples[-1,10]), file=f)
print('FWHM offset = {0}'.format(opt_samples[-1,11]), file=f)
print(file=f)
print('RVs jitter = {0}'.format(opt_samples[-1,12]), file=f)
print('BIS jitter = {0}'.format(opt_samples[-1,13]), file=f)
print('FWHM jitter = {0}'.format(opt_samples[-1,14]), file=f)
print('loglike= {0}'.format(opt_samples[-1,15]), file=f)
print(file=f)
print('done')

vals1, _, _ = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                                      jitters = jitters,time = time, dataset = 1)
vals2, _, _ = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                                      jitters = jitters,time = time, dataset = 2)
vals3, _, _ = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                                      jitters = jitters,time = time, dataset = 3)
residuals1 = val1 - vals1
rms1 = np.sqrt(np.sum(residuals1**2)/time.size)
residuals2 = val2 - vals2
rms2 = np.sqrt(np.sum(residuals2**2)/time.size)
residuals3 = val3 - vals3
rms3 = np.sqrt(np.sum(residuals3**2)/time.size)
print('RMS RVs (m/s):', rms1, file=f)
print('RMS BIS (m/s):', rms2, file=f)
print('RMS FWHM (m/s):', rms3, file=f)
print(file=f)
f.close()

fig, axs = plt.subplots(2,1, sharex=True, gridspec_kw={'width_ratios': [1],
                                                       'height_ratios': [3, 1]})
axs[0].set_title('Fit')
axs[0].set_ylabel('RV (m/s)')
axs[0].errorbar(time, val1, val1err, fmt= '.k')
axs[0].plot(tstar, mu11, '-r')
axs[1].set_title('Residuals (RMS:{0} m/s)'.format(np.round(rms1, 3)))
axs[1].set_ylabel('RMS (m/s)')
axs[1].plot(time, residuals1, '.k')
axs[1].axhline(y=0, linestyle='--', color='b')
axs[1].set_xlabel('Time (BJD - 2400000)')
plt.savefig('15_residualsPlot_RV.png')
plt.close('all')

fig, axs = plt.subplots(2,1, sharex=True, gridspec_kw={'width_ratios': [1],
                                                       'height_ratios': [3, 1]})
axs[0].set_title('Fit')
axs[0].set_ylabel('BIS (m/s)')
axs[0].errorbar(time, val2, val2err, fmt= '.k')
axs[0].plot(tstar, mu22, '-r')
axs[1].set_title('Residuals (RMS:{0} m/s)'.format(np.round(rms2, 3)))
axs[1].set_ylabel('RMS (m/s)')
axs[1].plot(time, residuals2, '.k')
axs[1].axhline(y=0, linestyle='--', color='b')
axs[1].set_xlabel('Time (BJD - 2400000)')
plt.savefig('16_residualsPlot_BIS.png')
plt.close('all')

fig, axs = plt.subplots(2,1, sharex=True, gridspec_kw={'width_ratios': [1],
                                                       'height_ratios': [3, 1]})
axs[0].set_title('Fit')
axs[0].set_ylabel('FWHM (m/s)')
axs[0].errorbar(time, val3, val3err, fmt= '.k')
axs[0].plot(tstar, mu33, '-r')
axs[1].set_title('Residuals (RMS:{0} m/s)'.format(np.round(rms3, 3)))
axs[1].set_ylabel('RMS (m/s)')
axs[1].plot(time, residuals3, '.k')
axs[1].axhline(y=0, linestyle='--', color='b')
axs[1].set_xlabel('Time (BJD - 2400000)')
plt.savefig('17_residualsPlot_FWHM.png')
plt.close('all')
