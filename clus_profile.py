


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.misc import imresize



def open_profile(filename):#ensure filename is a string
    PIX = 2048 # all files are 2048 x 2048
    fd = open(filename, 'rb')
    temp = np.fromfile(file=fd, dtype=np.float32)
    fd.close()
    profile = np.reshape(temp,(PIX,PIX))
    temp = 0
    #zoom = PIX/2 -128 - if we want the zoomed in cluster profile
    #profile=profile[int(zoom):-int(zoom),int(zoom):-int(zoom)] 
    return profile 

def radial_average(data,bins=300): #puts profile into annular bins
    centre_y= int((data.shape[0])/2)
    centre_x= int((data.shape[1])/2)
    y, x = np.indices((data.shape))
    r = np.sqrt((x - centre_x)**2 + (y - centre_y)**2)
    
    tbin, b1 =np.histogram(np.ravel(r), bins=bins, weights=np.ravel(data))
    nr, b2=np.histogram(np.ravel(r),bins=bins)
    index=nr==0
    nr[index]=1 #to avoid divide by zero
    radialprofile = tbin / nr
    return radialprofile

def rMatrix(bins=300): #gives a centred distance matrix
    
    y,x=np.indices((bins,bins))
    cent= int(bins/2)
    
    r = np.sqrt((x - cent)**2 + (y - cent)**2)
    return r

#issue: the profiles are square, the cmb maps are rectangular. 
# To make the averaged profile the same shape as the cmb map, I centred the square
# profile in an array of zeros that has the same shape of the cmb map.This seems wrong. 

def end_profile(rMatrix,radial_average,cmbmap, bins=300):
    x = np.arange(bins) 
    f = interp1d(x, radial_average)
    squareprofile= f(rMatrix.flat).reshape(rMatrix.shape)
    dim=np.min(cmbmap.shape)
    assert dim == cmbmap.shape[0]
    assert cmbmap.shape[1]%2 == 0
    diff=np.abs(cmbmap.shape[0]-cmbmap.shape[1])
    resized_prof=imresize(squareprofile,(dim,dim))
    profile = np.zeros((cmbmap.shape))
    
    if diff%2 == 0: #if difference is even 
        profile[:,int(diff/2):-int(diff/2)]=resized_prof
    else:
        profile[:,int(diff//2):-((int(diff//2) +1))]=resized_prof
    return profile


#opening 6 random cluster profiles
prof1= open_profile('GEN_Cluster_226L165.256.FBN2_snap44_comovFINE.d')
prof2= open_profile('GEN_Cluster_1L165.256.FBN2_snap35_comovFINE.d')
prof3= open_profile('GEN_Cluster_1L165.256.FBN2_snap39_comovFINE.d')
prof4= open_profile('GEN_Cluster_1L165.256.FBN2_snap50_comovFINE.d')
prof5= open_profile('GEN_Cluster_293L165.256.FBN2_snap40_comovFINE.d')
prof6= open_profile('GEN_Cluster_299L165.256.FBN2_snap52_comovFINE.d')


#Taking the element-wise average of the 6 profiles
average= np.mean(np.array([prof1,prof2,prof3,prof4,prof5,prof6]),axis=0)

#plt.imshow(average)

radial_ave= radial_average(average)

r= rMatrix()

cmbmap=np.zeros((1600,4000)) #array with shape of cmb map

end_prof= end_profile(r, radial_ave,cmbmap )

plt.imshow(end_prof)

