#First step lets import everything we need
import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy import stats
from artgpn.arttwo import network
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
weights = [weight.SquaredExponential(0.1, 20), weight.SquaredExponential(1.0, 20),
           weight.SquaredExponential(1.0, 20), weight.SquaredExponential(10.0, 20)]

#We are also defining our mean functions to be just constants/offset, even for the RVs
means = [mean.Constant(np.mean(val1)),
         mean.Constant(np.mean(val2)),
         mean.Constant(np.mean(val3)), 
         mean.Constant(np.mean(val4))]

#Finally we defined a term called jitter for all datasets
jitters =[np.std(val1), np.std(val2), np.std(val3), np.std(val4)]

#Having defined everything we now create the network we called GPnet
GPnet = network(1, time, val1, val1err, val2, val2err, val3, val3err, val4, val4err)

#Lets just check the likelihood of our model
loglike = GPnet.log_likelihood(nodes, weights, means, jitters)
print(loglike)