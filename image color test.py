#%% imports

import cv2, numpy as np, matplotlib.pyplot as plt, time, math, itertools
from numba import jit
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter

#%% extract data

def get_hsv(file):
    # get image
    img = cv2.imread(file,-1)
    # get basic image data
    H = img.shape[0]
    W = img.shape[1]
    img = img.reshape((H*W,4))
    # equalize transparent pixels and get unique values
    img[img[:,3]==0] = [0,0,0,0]
    img,inds,counts = np.unique(img,axis=0,return_counts=True,return_inverse=True)
    img = img.astype(float)
    # extract channels
    b = img[:,0] / 255
    g = img[:,1] / 255
    r = img[:,2] / 255
    a = img[:,3] / 255
    # solve for lightness
    Cmax = np.max([r,g,b],axis=0)
    Cmin = np.min([r,g,b],axis=0)
    c = Cmax-Cmin
    l = (Cmax+Cmin)/2
    # solve for saturation
    s = np.zeros_like(c,dtype=float)
    s[c!=0] = c[c!=0]/(1-np.abs(2*l[c!=0]-1))
    # solve for hue
    h = np.zeros_like(s,dtype=float)
    imax = np.argmax([r,g,b],axis=0)
    imax[s==0] = -1
    h[imax==0] = (((g[imax==0]-b[imax==0])/c[imax==0])%6)*(np.pi/3)
    h[imax==1] = (((b[imax==1]-r[imax==1])/c[imax==1])+2)*(np.pi/3)
    h[imax==2] = (((r[imax==2]-g[imax==2])/c[imax==2])+4)*(np.pi/3)
    h = (h + np.pi)%(2*np.pi) - np.pi
    # get cartisian for plot
    x = c*np.cos(h)
    y = c*np.sin(h)
    # remove irrelavent values
    h[s==0] = np.nan
    alpha0 = np.nonzero(a==0)[0][0]
    for val in ['counts','r','g','b','c','l','x','y','h','s','img','a']:
        locals()[val] = locals()[val][a!=0]
    L = l.shape[0]
    # save to important table
    data = np.array([x,y,l,a]).T
    color = np.array([r,g,b,a]).T
    return data,color,counts


#%% local max

def local_max(data,R=None,maxIter=100,R_val=None,weights=None):
    
    # get params
    dim = data.shape[1]
    R = np.ones(dim) if R is None else np.array(R)
    if R_val is not None: R *= R_val
    weights = np.ones(data.shape[1]) if weights is None else np.array(weights)
  
    # mean values rememberd
    K = 1/R
    cluster_mean = []
    cluster_log_mean = []

    # create grid array
    points = []
    for i in range(dim):
        s = R[i] * 2 / np.sqrt(dim)
        mx = np.max(data[:,i])
        mn = np.min(data[:,i])
        rg = mx-mn
        n = math.ceil(rg/s) 
        r = rg/n/2
        points.append(np.linspace(mn+r,mx-r,n))
    points = np.array([a.flatten() for a in np.meshgrid(*points)]).T    
   
    # solve for local maxima
    for point in points:
        dist = np.sum(((data-point)*K)**2,axis=1)
        inside = dist <= 1
        if not np.any(inside): continue
        mean = np.average(data[inside],weights=weights[inside],axis=0).tolist()
        mean_log = []
        count = 0
        
        while True:
            if mean in cluster_log_mean: break
            dist = np.sum(((data-mean)*K)**2,axis=1)
            inside = dist <= 1
            mean_old = mean
            mean = np.average(data[inside],weights=weights[inside],axis=0).tolist()
            count += 1
            mean_log.append(mean)
            if mean==mean_old or count>=maxIter:
                if count>=maxIter: print('Hit Max Iter')            
                cluster_mean.append(mean)
                break
        
        cluster_log_mean.extend(mean_log)
        
    return np.array(cluster_mean)

def window_max(A,W,maxIter=100,dev=50):

    s = np.array(A.shape)
    dim = len(s)
    if isinstance(W,int): W = np.ones(dim,dtype=int)*W
    rWm = ((W-1)/2).astype(int)
    rWp = ((W+1)/2).astype(int)
    
    # create grid array
    points = []
    for d in range(dim):
        points.append(np.arange(rWm[d],s[d],W[d],dtype=int))
    points = np.array([a.flatten() for a in np.meshgrid(*points)]).T    
    
    Wi = np.indices(W)
    
    cluster_mean = []
    cluster_log = []
    cluster_weights = []
    
    for p in points:
        
        p = p.tolist()
        if p in cluster_log: continue
    
        imin = np.clip(p-rWm,0,s-1)
        imax = np.clip(p+rWp,1,s)
        iran = np.insert(imax-imin,0,dim)
        a = A[tuple([slice(imin[d],imax[d]) for d in range(dim)])].flatten()
        if np.sum(a) == 0: continue
        
        p_log = [p]
        m_log = []
        w_log = []
        count = 1
        
        while True:
            
            imin = np.clip(p-rWm,0,s-1)
            imax = np.clip(p+rWp,1,s)
            iran = np.insert(imax-imin,0,dim)
            a = A[tuple([slice(imin[d],imax[d]) for d in range(dim)])].flatten()
            w = np.sum(a)
            if w == 0:
                break
            wi = Wi[tuple([slice(0,iran[d]) for d in range(dim+1)])].reshape((dim,-1))          
            m = np.average(wi,axis=1,weights=a) + p
            m_log.append(m)
            w_log.append(w)
            
            p = np.rint(m).astype(int).tolist()
            if p in cluster_log:
                cluster_log.extend(p_log)
                break
            if p in p_log or count >= maxIter:
                if count>=maxIter: print('Hit Max Iter')        
                i = p_log.index(p)
                cluster_mean.append(np.mean(m_log[i:],axis=0))
                cluster_weights.append(np.mean(w_log[i:],axis=0))
                cluster_log.extend(p_log)
                break
            p_log.append(p)
        
    m = np.array(cluster_mean)
    w = np.array(cluster_weights)
        
    m = m[w>np.max(w)/100,:]
    
    return m


    
def is_pos(dx):
    ans = np.zeros(dx.shape[0],dtype=bool)
    return is_pos1(dx,ans) 
@jit(nopython=True)
def is_pos1(dx,ans):
    N = len(ans)
    D = dx.shape[1]
    for n in range(N):
        for d in range(D-1,-1,-1):
            if dx[n,d] != 0:
                if dx[n,d] > 0: ans[n] = True
                break
    return ans
is_pos(np.array([[1,2,3]]))

def sum_bins(inds,weights):
    s = np.max(inds,axis=0)+1
    bins = np.zeros(s).flatten()
    inds = np.ravel_multi_index(inds.T,s)
    bins = sum_bins1(bins,inds,weights)
    bins = bins.reshape(s)
    return bins
@jit(nopython=True)
def sum_bins1(bins,inds,weights):
    for n in range(inds.shape[0]):
        bins[inds[n]] += weights[n]
    return bins
sum_bins(np.array([[0,2],[1,1]]),np.array([5,7]))
       
# def convolve(M1,M2):
#     sM1 = np.array(M1.shape)
#     sM2 = np.array(M2.shape)
#     sA = sM1 + sM2 -1
#     d = len(sA)
#     mA = np.array([np.product(sA[i+1:]).astype(int) for i in range(d)])
#     IM1 = np.indices(sM1).reshape((d,-1)).T
#     IM2 = np.indices(sM2).reshape((d,-1)).T
#     A = np.zeros(sA)
#     A = convolve1(A.flatten(),M1.flatten(),M2.flatten(),IM1,IM2,mA,d)
#     return A    
# @jit(nopython=True)
# def convolve1(A,M1,M2,IM1,IM2,mA,d):
#     LM1 = len(M1)
#     LM2 = len(M2)
#     for i1 in range(LM1):
#         for i2 in range(LM2):
#             ind = 0
#             for di in range(d):
#                 ind += mA[di]*(IM1[i1][di]+IM2[i2][di])
#             A[ind] += M1[i1]*M2[i2]
#     return A
# convolve(np.array([[1,2],[3,4]]),np.array([[1,1,1],[1,1,1],[1,1,1]]))


def gauss(x,s=1):
    return 1/s/np.sqrt(2*np.pi)*np.exp(-(x**2)/(2*(s**2)))

    
def gfilt(s):
    k = []
    for d in s:
        k.append(gauss(np.linspace(-3,3,d)))
    k = np.product(np.meshgrid(*k),axis=0)
    k /= gauss(0)**len(s)
    return k





def get_info(data,R=None,weights=None,R_val=None):
    
    dim = data.shape[1]
    R = np.ones(dim) if R is None else np.array(R)
    if R_val is not None: R = R*R_val
    K = 1/R
    
    weights = np.ones(data.shape[1]) if weights is None else np.array(weights)
    
    X = []
    DX = []
    W = []
    
    for p in range(data.shape[0]):
        dist = np.sum(((data-data[p])*K)**2,axis=1)
        inside = dist <= 1
        points = data[inside]       
        X.append(points)
        DX.append(data[p]-points)
        W.append(weights[inside]+weights[p])   
        
    X = np.vstack(X)
    DX = np.vstack(DX)
    W = np.concatenate(W)
    
    inds = is_pos(DX)
    X = X[inds]
    DX = DX[inds]
    W = W[inds]
    
    pj = np.sum(X*DX,axis=0)
    DX /= np.sum(DX**2,axis=0)
    Y = X + DX*pj

    L = np.hstack((Y,DX))

    clusters = local_max(L,R_val=.1,weights=W)


    # itertools.combinations
                
    return clusters
     
# clusters = local_max(data,R=[2,2,1,np.inf],mn=[-1,-1,0,0],R_val=0.02,weights=counts)    

# clusters = get_info(data[:,:3],R=[2,2,1],R_val=0.01,weights=counts)

def gen_sphere(n=None,da=10):
    r = (1/2)*(1+math.sqrt(5))
    points = [[ 0, 1, r],[ 0,-1, r],[ 0,-1,-r],[ 0, 1,-r],
              [ 1, r, 0],[-1, r, 0],[-1,-r, 0],[ 1,-r, 0],
              [ r, 0, 1],[-r, 0, 1],[-r, 0,-1],[ r, 0,-1]]
    points = np.array(points)
    points = (points.T / np.sqrt(np.sum(points**2,axis=1))).T
    edges = get_edges(points)
    if n is None: n = math.ceil(np.log2(np.arccos(1/np.sqrt(5))*180/np.pi/da))
    for _ in range(n):
        new_points = points.tolist()       
        for i1,i2 in edges:
            point = ( points[i1] + points[i2] ) / 2
            point /= np.sqrt(np.sum(point**2))
            new_points.append(point)
        points = np.array(new_points)
        edges = get_edges(points)
    return points
def get_edges(points):
    edges = []
    for p,point in enumerate(points):
        dist = np.sum((points-point)**2,axis=1)
        inds = np.nonzero(dist<np.min(dist[dist>0])*1.5)[0]
        inds = inds[inds>p]
        edges.extend([p,ind] for ind in inds)
    return edges

def equal_ax(ax):
    X,Y,Z = [],[],[]
    for line in ax.lines:
        x,y,z = line._verts3d
        X.extend(x.tolist())
        Y.extend(y.tolist())
        Z.extend(z.tolist())
    ax.set_box_aspect((np.ptp(X),np.ptp(Y),np.ptp(Z)))






def d_hough(data,dx=0.1):

    pass
    


file = 'Bed, medium, double.png'
data,color,weights = get_hsv(file)
data = data[:,0:3]

dx = 0.1
g = 10
ptp = np.sqrt(np.sum(np.ptp(data)**2))
da = np.arcsin(dx/(ptp/2))*180/np.pi
B = gen_sphere(da=da)
B = B[is_pos(B)]
bx = B[:,0]; by = B[:,1]; bz = B[:,2]
cx = np.array([1-(bx**2)/(1+bz),-(bx*by)/(1+bz),-bx])
cy = np.array([-(bx*by)/(1+bz),1-(by**2)/(1+bz),-by])

B = np.vstack((B,-B))
cx = np.hstack((cx,-cx))
cy = np.hstack((cy,-cy))


W = np.repeat(weights,B.shape[0])
B = np.tile(B.T,data.shape[0]).T
X = np.dot(data,cx).reshape((B.shape[0],1))
Y = np.dot(data,cy).reshape((B.shape[0],1))

D = np.hstack((B,X,Y))
ds = np.sin(da)/2
d = np.array([ds,ds,ds,dx,dx])/g
m = np.min(D,axis=0)
I = np.floor((D-m)/d).astype(int)

A = sum_bins(I,W)
# G = gfilt([g*3]*I.shape[1])
# A = convolve(A,G)
# A = gaussian_filter(A,2)
# ans = window_max(A,g*2+1)
# ans = ans*d + m + d/2




#%% cluster

#     dist = np.sum(((data-mean)*K)**2,axis=1)
#     inside = dist <= 1
#     mean = np.average(data[inside],weights=counts[inside],axis=0).tolist()

# for corner in start:
    
#     ind = np.logical_and(np.all(data>=corner,axis=1),np.all(data<corner+2*R,axis=1))
#     inds = np.nonzero(ind)[0].tolist()
    
#     if len(inds) > 0:
    
#         mean = (corner+R).tolist()    
#         mean_old = mean
#         mean_log = []
#         count = 0
#         while True:
#             try:
#                 ind_c = cluster_log_ind[cluster_log_mean.index(mean)]
#                 cluster_inds[ind_c].extend(inds)
#                 break
#             except ValueError: pass
                
#             dist = np.sum(((data-mean)*K)**2,axis=1)
#             inside = dist <= 1
#             mean = np.average(data[inside],weights=counts[inside],axis=0).tolist()
#             count += 1
#             mean_log.append(mean)
            
#             if mean==mean_old or count>=maxIter:
#                 if count>=maxIter:
#                     print('Hit Max Iter')            
#                 cluster_mean.append(mean)
#                 cluster_inds.append(inds)
#                 ind_c = len(cluster_mean)-1
#                 break
    
#         cluster_log_mean.extend(mean_log)
#         cluster_log_ind.extend([ind_c]*len(mean_log))   


# for i in range(L):
#     mean = data[i,:].tolist()
    
#     dist = np.sum(((data-mean)*K)**2,axis=1)
#     inside = dist <= 1
#     mean = np.average(data[inside],weights=counts[inside],axis=0).tolist()
    
#     mean = data[i,:].tolist()
#     mean_old = mean
#     mean_log = []
#     count = 0
    
#     while True:
#         try:
#             ind_c = cluster_log_ind[cluster_log_mean.index(mean)]
#             cluster_inds[ind_c].append(i)
#             break
#         except ValueError: pass
            
#         dist = np.sum(((data-mean)*K)**2,axis=1)
#         inside = dist <= 1
#         mean = np.average(data[inside],weights=counts[inside],axis=0).tolist()
#         count += 1
        
#         try:
#             ind_m = mean_log.index(mean)
#             print(f'Hit Mean Loop Size {len(mean_log)-ind_m+1}')
#             mean = np.average(mean_log[ind_m:],axis=0).tolist()
#             cluster_mean.append(mean)
#             cluster_inds.append([i])
#             ind_c = len(cluster_mean)-1
#             break
#         except ValueError: pass
        
#         mean_log.append(mean)
        
#         if mean==mean_old or count>=maxIter:
#             if count>=maxIter:
#                 print('Hit Max Iter')            
#             cluster_mean.append(mean)
#             cluster_inds.append([i])
#             ind_c = len(cluster_mean)-1
#             break


#     cluster_log_mean.extend(mean_log)
#     cluster_log_ind.extend([ind_c]*len(mean_log))   
    
# print(f'Time: {time.time()-t}')

#%% 

# clusters = []
# c_colors = []
# colors = ['']*len(data)
# for i,(mean,inds) in enumerate(zip(cluster_mean,cluster_inds)):
#     dist = np.sum(((data-mean)*K)**2,axis=1)
#     inside = dist <= 1
#     clusters.append(np.average(data[inside],weights=counts[inside]**5,axis=0).tolist())
#     for ind in inds:
#         colors[ind] = f'C{i}'
#     c_colors.append(f'C{i}')
# clusters = np.array(clusters)


#%% plot


# # make bounding box
# theta = np.linspace(0,2*np.pi,num=6*5,endpoint=False)
# x_ = np.cos(theta)
# y_ = np.sin(theta)
# z_ = np.ones_like(theta)*.5
# x_ = np.append(x_,0)
# y_ = np.append(y_,0)
# z_ = np.append(z_,1)

# # plot
# plt.close('all')
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# size = np.clip( weights / np.mean(weights) , .1, 10)
# color[:,-1] = 1
# ax.scatter(data[:,0],data[:,1],data[:,2],s=size,c=color)
# ax.plot_trisurf(x_,y_,z_,color=[.5,.5,.5,.1])
# z_[-1] = 0
# ax.plot_trisurf(x_,y_,z_,color=[.5,.5,.5,.1])

# bx = ans[:,0]; by = ans[:,1]; bz = ans[:,2]; x = ans[:,3]; y = ans[:,4]
# cx = np.array([1-(bx**2)/(1+bz),-(bx*by)/(1+bz),-bx])
# cy = np.array([-(bx*by)/(1+bz),1-(by**2)/(1+bz),-by])
# p1 = (cx*x + cy*y).T
# p2 = p1 + ans[:,0:3]
# P = np.stack((p1,p2),axis=-1)
# for p in P:
#     p = p.T
#     ax.plot(p[:,0],p[:,1],p[:,2])
    
    

# equal_ax(ax)

# ax.scatter(clusters[:,0],clusters[:,1],clusters[:,2],s=50,c='r')