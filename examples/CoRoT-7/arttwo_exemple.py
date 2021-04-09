#First step lets import everything we need
import numpy as np
import matplotlib.pyplot as plt

from artgpn.arttwo import network
from artgpn import weight, node, mean

#We also need data to work with in this case data of CoRoT-7
time,rv,rverr,fwhm,fwhmerr,bis,biserr,rhk,rhkerr = np.loadtxt("corot7_data.rdb", 
                                                              skiprows=1,max_rows=20, 
                                                              unpack=True, 
                                                              usecols=(0,1,2,3,4,5,6,7,8))

#  To make our life easier we are renaming and standardizing some datasets, but
#you dont need to do it
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

#    Having the data we now choose our nodes and weights
#    A Periodic kernel for a node and a Squared Exponential kernel as weight,
#this allow, for example, to fit all the data were they have the same parameters
#of the node (the period in this case), but the datasets will have a different
#amplitude and decaying timescale that is given on the weights
nodes = [node.Periodic(20, 0.5)] 
weights = [weight.SquaredExponential(10.0, 1.0), 
           weight.SquaredExponential(2.0, 5.0),
           weight.SquaredExponential(3.0, 20),
           weight.SquaredExponential(5.0, 30)]
#    The first weight fits the RVs, the second will go for the FWHM, third BIS,
#and the last will fit the Rhk.
#    If, for example, you believed all your dataset had the same period, and
#decaying timescales but different amplitudes you could then define a 
#quasiperiodic kernel on the node (node.QuasiPeriodic) and make the weights 
#constants (weight.Constant)

#    We are also defining our mean functions to be just constants/offset, there
#are others availables
means = [mean.Constant(np.mean(val1)),
         mean.Constant(np.mean(val2)),
         mean.Constant(np.mean(val3)), 
         mean.Constant(np.mean(val4))]

#    Finally we defined a term called jitter for all datasets
jitters =[np.std(val1), np.std(val2), np.std(val3), np.std(val4)]

#    Having defined everything we now create the network we called GPnet
GPnet = network(1, time, val1, val1err, val2, val2err, val3, val3err, val4, val4err)
#    The 1 represents the number of node you have, in our example 1 node.Periodic,
#but you could define 2, 3 or 4, just dont forget more nodes will require more
#weights, that is, if you have P nodes and Q datasets then you need PxQ weights.

#    Lets now check the likelihood of our model
loglike = GPnet.log_likelihood(nodes, weights, means, jitters)
print(loglike)

#    If you are used to work with GPs from here on it easy, you can use this 
#packcage together with emcee for example you use an mcmc to find the best 
#solutions

#    For last ytou can of course plot the fits, lets see if looks good,
extention = 5
mu11, std11, cov11 = GPnet.prediction(nodes = nodes, weights = weights, 
                                      means = means,
                                      jitters = jitters,
                                      time = np.linspace(time.min()-extention, 
                                                         time.max()+extention,
                                                         5000),
                                      dataset = 1)
mu22, std22, cov22 = GPnet.prediction(nodes = nodes, weights = weights, 
                                      means = means,
                                      jitters = jitters,
                                      time = np.linspace(time.min()-extention, 
                                                         time.max()+extention,
                                                         5000),
                                      dataset = 2)
mu33, std33, cov33 = GPnet.prediction(nodes = nodes, weights = weights, 
                                      means = means,
                                      jitters = jitters,
                                      time = np.linspace(time.min()-extention, 
                                                         time.max()+extention,
                                                         5000),
                                      dataset = 3)
mu44, std44, cov44 = GPnet.prediction(nodes = nodes, weights = weights, 
                                      means = means,
                                      jitters = jitters,
                                      time = np.linspace(time.min()-extention, 
                                                         time.max()+extention,
                                                         5000),
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

