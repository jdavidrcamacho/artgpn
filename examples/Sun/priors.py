import numpy as np
time,rv,rverr,bis,biserr,fwhm,fwhmerr= np.loadtxt("sun50points.txt",
                                                  skiprows = 1, unpack = True,
                                                  usecols = (0,1,2,7,8,9,10))
val1, val1err, val2,val2err, val3, val3err = rv, rverr, bis, biserr, fwhm, fwhmerr

##### Setting priors #####
from scipy import stats
from loguniform import ModifiedLogUniform
stats.loguniform = stats.reciprocal

#node function
neta1 = stats.loguniform(0.1, 2*val1.ptp())
neta2 = stats.loguniform(np.ediff1d(time).mean(), time.ptp())
neta3 = stats.uniform(10, 50 -10)
neta4 = stats.loguniform(0.1, 5)

#weight function
weta1_1 = stats.loguniform(0.1, 2*val1.ptp())
weta2_1 = stats.loguniform(np.ediff1d(time).mean(), 10*time.ptp())
weta1_2 = stats.loguniform(0.1, 2*val2.ptp())
weta2_2 = stats.loguniform(np.ediff1d(time).mean(), 10*time.ptp())
weta1_3 = stats.loguniform(0.1, 2*val3.ptp())
weta2_3 = stats.loguniform(np.ediff1d(time).mean(), 10*time.ptp())

#Mean function
#(1/pi)*(1/(1+slope*slope)) 
slope1 = stats.norm(0, 1)
offset1 = stats.uniform(val1.min(), val1.max() -val1.min())
slope2 = stats.norm(0, 1)
offset2 = stats.uniform(val2.min(), val2.max() -val2.min())
slope3 = stats.norm(0, 1)
offset3 = stats.uniform(val3.min(), val3.max() -val3.min())

#Jitter
jitt1 = ModifiedLogUniform(0.1, 2*val1.ptp())
jitt2 = ModifiedLogUniform(0.1, 2*val2.ptp())
jitt3 = ModifiedLogUniform(0.1, 2*val3.ptp())


def priors():
    return np.array([neta1, neta2, neta3, neta4, 
                     weta1_1, weta2_1, weta1_2, weta2_2, weta1_3, weta2_3, 
                     slope1, offset1, slope2, offset2, slope3, offset3, 
                     jitt1, jitt2, jitt3])
