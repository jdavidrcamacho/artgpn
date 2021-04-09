#for multiprocessing
import os
os.environ["OMP_NUM_THREADS"] = "1"
ncpu  = 4

import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 5]
plt.close('all')

import corner
labels = np.array(["eta3", "eta4", 
                   "weight11", "weight21", "weight12", "weight22",
                   "weight13", "weight23",
                   "slope1", "offset1", "slope2", "offset2", "slope3", "offset3",
                   "jitter1", "jitter2", "jitter3", "logl"])
import emcee
max_n = 1000000 #defining iterations

from artgpn.arttwo import network
from artgpn import weight, node, mean

time,rv,rverr,bis,biserr, fw, fwerr = np.loadtxt("sun50points.txt",
                                                 skiprows = 1, unpack = True, 
                                                 usecols = (0,1,2,7,8,9,10))
time, val1, val1err = time, rv, rverr
val2,val2err = bis, biserr
val3,val3err = fw, fwerr

plt.rcParams['figure.figsize'] = [15, 3*5]
fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
axs[0].errorbar(time, val1, val1err, fmt='.k')
axs[0].set_ylabel('RV (m/s)')
axs[1].errorbar(time, val2, val2err, fmt='.k')
axs[1].set_ylabel('BIS (m/s)')
axs[2].errorbar(time, val3, val3err, fmt='.k')
axs[2].set_xlabel('Time (BJD-2400000)')
axs[2].set_ylabel('FWHM (m/s)')
plt.savefig('10_dataset.png')
plt.close('all')

nodes = [node.QuasiPeriodic(10, 20, 0.5)] 
weights = [weight.Constant(1), weight.Constant(1), weight.Constant(1)]
means = [mean.Linear(0, np.mean(val1)),
         mean.Linear(0, np.mean(val2)),
         mean.Linear(0, np.mean(val3))]
jitters =[np.std(val1), np.std(val2), np.std(val3)]

#Having defined everything we now create the network we called GPnet
GPnet = network(1, time, val1, val1err, val2, val2err, val3, val3err)
loglike = GPnet.log_likelihood(nodes, weights, means, jitters)
print(loglike)

from scipy import stats
stats.loguniform = stats.reciprocal
from priors import priors
prior = priors()

def prior_transform():
    per = prior[2].rvs()
    new_eta2 = stats.loguniform(0.5*per, time.ptp())
    return np.array([new_eta2.rvs(), per, prior[3].rvs(), 
                     prior[4].rvs(), prior[6].rvs() ,prior[8].rvs(),
                     prior[10].rvs(), prior[11].rvs(),
                     prior[12].rvs(), prior[13].rvs(),
                     prior[14].rvs(), prior[15].rvs(),
                     prior[16].rvs(), prior[17].rvs(), prior[18].rvs()])

#log_transform calculates our posterior
def log_transform(theta):
    n2,n3,n4,w11,w12,w13,s1,off1,s2,off2,s3,off3,j1,j2,j3 = theta
    if n2 < 0.5*n3:
        return -np.inf
        
    logcdfEta2 = prior[1].logcdf(0.5*n3)
    if np.isinf(logcdfEta2):
        return -np.inf
        
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
    if np.isinf(logprior):
        return -np.inf

    nodes = [node.QuasiPeriodic(n2, n3, n4)]
    weights = [weight.Constant(w11), weight.Constant(w12), weight.Constant(w13)]
    means = [mean.Linear(s1, off1), mean.Linear(s2, off2), mean.Linear(s3, off3)]
    jitters = [j1, j2, j3]

    logpost = logprior + GPnet.log_likelihood(nodes, weights, means, jitters)
    return logpost
##### Sampler definition #####
ndim = prior_transform().size
nwalkers = 2*ndim

#Set up the backend
filename = "savedProgress.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

from multiprocessing import Pool
pool = Pool(ncpu)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_transform, 
                                backend=backend, pool=pool)

#Initialize the walkers
p0=[prior_transform() for i in range(nwalkers)]
print("\nRunning...")

#We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)
#This will be useful to testing convergence
old_tau = np.inf
#Now we'll sample for up to max_n steps
for sample in sampler.sample(p0, iterations=max_n, progress=True):
    # Only check convergence every 100 steps
    if sampler.iteration % 5000:
        continue
    #Compute the autocorrelation time so far
    tau = sampler.get_autocorr_time(tol=0)
    burnin = int(2*np.max(tau))
    thin = int(0.5 * np.min(tau))
    autocorr[index] = np.mean(tau)
    index += 1
    # Check convergence
    converged = np.all(tau * 25 < sampler.iteration)
    #converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau
    #plotting corner
    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
    #log_prob_samples = np.nan_to_num(log_prob_samples)
    print("burn-in: {0}".format(burnin))
    print("thin: {0}".format(thin))
    print("flat chain shape: {0}".format(samples.shape))
    all_samples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)
    corner.corner(all_samples[:,:], labels=labels, color="k", bins = 50,
                  quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True, 
                  show_titles=True, plot_density=True, plot_contours=True,
                  fill_contours=True, plot_datapoints=False)
    plt.savefig('tmp_corner_{0}.png'.format(sampler.iteration))
    plt.close('all')


##### Plots #####
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.5 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
all_samples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)

labels = np.array(["eta2", "eta3", "eta4", 
                   "weight1", "weight2", "weight3",
                   "slope1", "offset1", "slope2", "offset2", "slope3", "offset3", 
                   "jitter1", "jitter2", "jitter3"])
corner.corner(all_samples[:, :-1], labels=labels, color="k", bins = 50,
              quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True, 
              show_titles=True, plot_density=True, plot_contours=True,
              fill_contours=True, plot_datapoints=False)
plt.savefig('02_cornerPlot.png')
plt.close('all')

labels = np.array(["eta2", "eta3", "eta4", 
                   "weight1", "weight2", "weight3",
                   "slope1", "offset1", "slope2", "offset2", "slope3", "offset3", 
                   "jitter1", "jitter2", "jitter3",
                   "log prob"])
corner.corner(sampler.get_chain(flat=True), labels=labels, color="k", bins = 50,
              quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True, 
              show_titles=True, plot_density=True, plot_contours=True,
              fill_contours=True, plot_datapoints=True)
plt.savefig('03_cornerPlot.png')
plt.close('all')

