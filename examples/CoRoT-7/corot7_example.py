#First step lets import everything we need
import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy import stats
from artgpn.art import network
from artgpn import weight, node, mean, utils

#We also need data to work with in this case data of CoRoT-7
time,rv,rverr,fwhm,fwhmerr,bis,biserr,rhk,rhkerr = np.loadtxt("corot7_data.rdb", 
                                                              skiprows=1, 
                                                              unpack=True, 
                                                              usecols=(0,1,2,3,4,5,6,7,8))

#To make our life easier we are renaming and standardizing some datasets
val1 = rv #RVs
val1err = rverr
val2 = fwhm #FWHM
val2err = fwhmerr
val2 -= val2.mean() 
val2err /= val2.std()
val2 /= val2.std()
val3 = bis #BIS
val3err = biserr
val3 -= val3.mean() 
val3err /= val3.std()
val3 /= val3.std()
val4 = rhk #log R'hk
val4err = rhkerr
val4 -= val4.mean() 
val4err /= val4.std()
val4 /= val4.std()

#Having the data we now choose our nodes and weights
#A Periodic kernel for a node and a Squared Exponential kernel as weight
nodes = [node.Periodic(20, 0.5)] 
weights = [weight.SquaredExponential(1.0, 20)]
#Due the way its implemented we need to give amplitudes separately to our weights
weight_amplitudes = [1.0, 1.0, 1.0, 1.0]

#We are also defining our mean functions to be just constants/offset, even for the RVs
means = [mean.Constant(np.mean(val1)),
         mean.Constant(np.mean(val2)),
         mean.Constant(np.mean(val3)), 
         mean.Constant(np.mean(val4))]

#Finally we defined a term called jitter for all datasets
jitters =[np.std(val1), np.std(val2), np.std(val3), np.std(val4)]

#Having defined everything we now create the network we called GPnet
GPnet = network(nodes, weights, weight_amplitudes, means, jitters, time, 
                  val1, val1err, val2, val2err, val3, val3err, val4, val4err)

#Lets just check the likelihood of our model
loglike = GPnet.log_likelihood(nodes, weights, weight_amplitudes, means, jitters)
print(loglike)

import sys
sys.exit(0)

#Now that we have a network we can run a MCMC to optimize our parameters
#We will start with creating our priors 
#node function priors
eta3_1 = stats.uniform(10, 40 -10)
eta4_1 = stats.uniform(0, 5 -0)
s = stats.uniform(0, 2 -0)

#weight function priors
eta2 = stats.uniform(np.ediff1d(time).mean(), 2*time.ptp() -np.ediff1d(time).mean())
#weight function amplitude for the RVs
w11 = stats.uniform(0, 5*val1.ptp() -0)
w12 = stats.uniform(0, 5*val1.ptp() -0)
#weight function amplitude for the FWHM
w21 = stats.uniform(0, 5*val2.ptp() -0)
w22 = stats.uniform(0, 5*val2.ptp() -0)
#weight function amplitude for the BIS
w31 = stats.uniform(0, 5*val3.ptp() -0)
w32 = stats.uniform(0, 5*val3.ptp() -0)
#weight function amplitude for the Rhk
w41 = stats.uniform(0, 5*val4.ptp() -0)
w42 = stats.uniform(0, 5*val4.ptp() -0)

#mean functions priors
mean_c1 = stats.uniform(5*val1.min(), 5*val1.max() -5*val1.min())
mean_c2 = stats.uniform(5*val2.min(), 5*val2.max() -5*val2.min())
mean_c3 = stats.uniform(5*val3.min(), 5*val3.max() -5*val3.min())
mean_c4 = stats.uniform(5*val4.min(), 5*val4.max() -5*val4.min())

#jitters priors
jitt1 = stats.uniform(0, 2*val1.std() -0)
jitt2 = stats.uniform(0, 2*val2.std() -0)
jitt3 = stats.uniform(0, 2*val3.std() -0)
jitt4 = stats.uniform(0, 2*val4.std() -0)


#The MCMC we will use is called emcee and required two functions
#prior_transform calls our priors
def prior_transform():
    return np.array([eta2.rvs(), eta3_1.rvs(), eta4_1.rvs(), s.rvs(), \
                     w11.rvs(), w21.rvs(), w31.rvs(), w41.rvs(), \
                     mean_c1.rvs(), mean_c2.rvs(), mean_c3.rvs(), mean_c4.rvs(), \
                     jitt1.rvs(), jitt2.rvs(), jitt3.rvs(), jitt4.rvs()])

#log_transform calculates our posterior
def log_transform(theta):
    n2, n31, n41, s1, weight11, weight21, weight31, weight41, \
    c1, c2, c3, c4, j1, j2, j3, j4 = theta
    
    logprior = eta2.logpdf(n2) + eta3_1.logpdf(n31) + eta4_1.logpdf(n41) +\
                w11.logpdf(weight11) + w21.logpdf(weight21) +\
                w31.logpdf(weight31) + w41.logpdf(weight41) +\
                mean_c1.logpdf(c1) + mean_c2.logpdf(c2) +\
                mean_c3.logpdf(c3) + mean_c4.logpdf(c4) +\
                jitt1.logpdf(j1) + jitt2.logpdf(j2) +\
                jitt3.logpdf(j3) + jitt4.logpdf(j4)

    new_nodes = [node.Periodic(n31, n41, s1)]
    
    #notice that our new_weights has an amplitude of "1.0"
    #This is because the amplitudes are defines separately
    new_weights = [weight.SquaredExponential(1.0, n2)]
    
    new_weights_amplitudes = [weight11, weight21, weight31, weight41]
    
    new_mean = [mean.Constant(c1), mean.Constant(np.mean(c2)),
                mean.Constant(np.mean(c3)), mean.Constant(np.mean(c4))]
    
    new_jitters = [j1, j2, j3, j4]
    return logprior + GPnet.log_likelihood(new_nodes, new_weights, 
                                           new_weights_amplitudes, new_mean, 
                                           new_jitters)
    
#In the utils we have a function to run our MCMC that can be defined as
samples = utils.run_mcmc(prior_transform, log_transform, iterations = 10000, 
                         sampler = 'emcee')

#When our MCMC is finished we can check the results
#Of the 10000 iterations, 5000 are burn-ins
#We then need to calculate median and quantiles and then print them
n2, n31, n41, s, w11, w21, w31, w41, \
c1,c2,c3,c4, j1,j2,j3,j4, logl = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))
print()
print('eta 2= {0[0]} +{0[1]} -{0[2]}'.format(n2))
print('eta 3= {0[0]} +{0[1]} -{0[2]}'.format(n31))
print('eta 4= {0[0]} +{0[1]} -{0[2]}'.format(n41))
print('s= {0[0]} +{0[1]} -{0[2]}'.format(s))
print()
print('weight 11= {0[0]} +{0[1]} -{0[2]}'.format(w11))
print('weight 21= {0[0]} +{0[1]} -{0[2]}'.format(w21))
print('weight 31= {0[0]} +{0[1]} -{0[2]}'.format(w31))
print('weight 41= {0[0]} +{0[1]} -{0[2]}'.format(w41))
print()
print('RVs offset = {0[0]} +{0[1]} -{0[2]}'.format(c1))
print('FWHM offset = {0[0]} +{0[1]} -{0[2]}'.format(c2))
print('BIS offset = {0[0]} +{0[1]} -{0[2]}'.format(c3))
print('Rhk offset = {0[0]} +{0[1]} -{0[2]}'.format(c4))
print()
print('RVs jitter = {0[0]} +{0[1]} -{0[2]}'.format(j1))
print('FWHM jitter = {0[0]} +{0[1]} -{0[2]}'.format(j2))
print('BIS jitter = {0[0]} +{0[1]} -{0[2]}'.format(j3))
print('Rhk jitter = {0[0]} +{0[1]} -{0[2]}'.format(j4))
print()

#We can also make a corner plot of our solution
corner.corner(samples, labels=["eta2", "eta3_1", "eta4_1", "s", "w11", "w21", 
                               "w31", "w41", "RV offset", "FWHM offset", 
                               "BIS offset", "Rhk offset", "RV jitter", 
                               "FWHM jitter", "BIS jitter", "Rhk jitter", 
                               "loglike"],
              show_titles=True, plot_contours = True, 
              plot_density = True, plot_points = False)

#Having a solution from our MCMC we need to redefine all the network
nodes = [node.Periodic(n31[0], n41[0], s[0])]
weights = [weight.SquaredExponential(1.0, n2[0])]
weights_values = [w11[0], w21[0], w31[0], w41[0]]
means = [mean.Constant(c1[0]), mean.Constant(c2[0]),
         mean.Constant(c3[0]), mean.Constant(c4[0])]
jitters = [j1[0], j2[0],j3[0], j4[0]]

GPnet = network(nodes, weights, weights_values, means, jitters, time, 
                  val1, val1err, val2, val2err, val3, val3err, val4, val4err)

#And then plot our fit to see if looks good
extention = 5
mu11, std11, cov11 = GPnet.prediction(nodes = nodes, weights = weights, 
                                      weights_values = weights_values, means = means,
                                      jitters = jitters,
                                      time = np.linspace(time.min()-extention, time.max()+extention, 5000),
                                      dataset = 1)
mu22, std22, cov22 = GPnet.prediction(nodes = nodes, weights = weights, 
                                      weights_values = weights_values, means = means,
                                      jitters = jitters,
                                      time = np.linspace(time.min()-extention, time.max()+extention, 5000),
                                      dataset = 2)
mu33, std33, cov33 = GPnet.prediction(nodes = nodes, weights = weights, 
                                      weights_values = weights_values, means = means,
                                      jitters = jitters,
                                      time = np.linspace(time.min()-extention, time.max()+extention, 5000),
                                      dataset = 3)
mu44, std44, cov44 = GPnet.prediction(nodes = nodes, weights = weights, 
                                      weights_values = weights_values, means = means,
                                      jitters = jitters,
                                      time = np.linspace(time.min()-extention, time.max()+extention, 5000),
                                      dataset = 4)
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
ax1.set_title('Fits')
ax1.fill_between(np.linspace(time.min()-extention, time.max()+extention, 5000), 
                 mu11+std11, mu11-std11, color="grey", alpha=0.5)
ax1.plot(np.linspace(time.min()-extention, time.max()+extention, 5000), mu11, "k--", alpha=1, lw=1.5)
ax1.errorbar(time,val1, val1err, fmt = "b.")
ax1.set_ylabel("RVs")

ax2.fill_between(np.linspace(time.min()-extention, time.max()+extention, 5000), 
                 mu22+std22, mu22-std22, color="grey", alpha=0.5)
ax2.plot(np.linspace(time.min()-extention, time.max()+extention, 5000), mu22, "k--", alpha=1, lw=1.5)
ax2.errorbar(time,val2, val2err, fmt = "b.")
ax2.set_ylabel("FWHM")

ax3.fill_between(np.linspace(time.min()-extention, time.max()+extention, 5000), 
                 mu33+std33, mu33-std33, color="grey", alpha=0.5)
ax3.plot(np.linspace(time.min()-extention, time.max()+extention, 5000), mu33, "k--", alpha=1, lw=1.5)
ax3.errorbar(time,val3, val3err, fmt = "b.")
ax3.set_ylabel("BIS")

ax4.fill_between(np.linspace(time.min()-extention, time.max()+extention, 5000), 
                 mu44+std44, mu44-std44, color="grey", alpha=0.5)
ax4.plot(np.linspace(time.min()-extention, time.max()+extention, 5000), mu44, "k--", alpha=1, lw=1.5)
ax4.errorbar(time,val4, val4err, fmt = "b.")
ax4.set_ylabel("Rhk")
plt.show()