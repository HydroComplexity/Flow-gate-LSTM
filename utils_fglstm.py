import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.stats import gaussian_kde
import torch

def nan_helper(y) :
    """Helper to handle indices and logical indices of NaNs.
    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
        to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return (np.isnan (y), lambda z : z.nonzero ()[0])


def intp(X) :
    training_set0 = np.ones (shape=X.shape)
    xx = X.astype (np.float)
    for i in range (X.shape[1]) :
        xx1 = xx[:, i]
        nans, yy = nan_helper (xx1)
        xx1[nans] = np.interp (yy (nans), yy (~nans), xx1[~nans])
        training_set0[:, i] = xx1
    return training_set0


def dataretrive(vari, inputvar,nk) :
    training_set = pd.read_excel ('//home/.../sd01.xlsx', sheet_name="7hour edited data",header=0)
    training_set1 = training_set.iloc[3646 :-nk, [vari]].values.  #upper hafren data start at 3646
    gi, gf = 504, 2200 # lot of missing data untill 500 data points 
    training_set2 = training_set.iloc[3646 :-nk, [inputvar[0]]].values
    for j in range (len (inputvar)) :
        training_set2 = np.hstack ((training_set2, training_set.iloc[3646 :-nk, [inputvar[j]]].values))

    training_set2= training_set2[:,1:]
    training_set2 = np.hstack ((training_set1[gi :gf], training_set2[gi :gf]))

    sc1 = MinMaxScaler ()
    training_set2 = intp (training_set2)
    training_set2 = sc1.fit_transform (training_set2)
    return training_set2

def getgrad(x):
    y=np.ones((len(x)-1))
    for i in range(len(y)):
        y[i]=x[i+1]-x[i]
    return np.array(y)

def sliding_windows(data, seq_length) :
    x = []
    y = []

    for i in range (len (data) - seq_length - 1) :
        _x = data[i :(i + seq_length)]
        _y = data[i + seq_length]
        x.append (_x)
        y.append (_y)

    return np.array (x), np.array (y)

def lstmtrain(num_epochs,seq,optimizer,criterion,input,target,lss,k, return_dict):
    for i in range (num_epochs) :
        seq.train()
        optimizer.zero_grad()
        out = seq(input)  # trainX      data x sequencelenth x number of variable
        loss = criterion(out, target)  # training set target=trainY
        if i % 2 == True :
            print ('epoch:', i, 'loss:', loss.item())
        if loss.item () <= .0005 :
            break
        lss[i] = loss.item ()
        if i >= 3 :
            lss1 = lss[i] - lss[i - 3]
            if abs (lss1) <= 0.0003 :
                break
        loss.backward()
        optimizer.step()

    out = seq(input)
    loss = criterion (out, target)  # training set target=trainY
    # if i % 1 == True :
    print ('after training: ', 'loss:', loss.item ())
    return_dict[k] = seq

def nse(predictions, targets) :
    return (1 - (((predictions - targets) ** 2) / ((targets - np.mean (targets)) ** 2)))


def mutualinfo(X,i):
    # kde of x1,x2 and joint x1,x2
    bin=15
    epsl=.000000001

    x,y=X[:, -1],X[:, -1-i]

    hmin,hmax=np.min(x),np.max(x)
    vmin,vmax=np.min(y),np.max(y)
    grid_bvar1 = np.linspace (hmin, hmax, bin)
    grid_bvar2 = np.linspace (vmin, vmax, bin)

    v1 = gaussian_kde (x)
    v1_kde = v1 (grid_bvar1)
    p1_kde = v1_kde / sum (v1_kde)
    v2 = gaussian_kde (y)
    v2_kde = v2 (grid_bvar2)
    p2_kde = v2_kde / sum (v2_kde)

    X1, Y1 = np.meshgrid (grid_bvar1, grid_bvar2)
    xf = np.vstack ([x, y])
    P_xy = gaussian_kde (xf)
    kas = np.vstack ([X1.ravel (), Y1.ravel ()])
    Z = np.reshape (P_xy (kas).T, X1.shape)
    Z1 = Z / sum (sum (Z))

    p1_kde[p1_kde == 0] = epsl
    p2_kde[p2_kde == 0] = epsl
    Z1[Z1 == 0] = epsl
            # entropy calculations
    hxy=-Z1 * np.log2(Z1)
    hx=-p1_kde * np.log2(p1_kde)
    hy=-p2_kde * np.log2(p2_kde)
    h_x1_x2 = sum(sum(hxy))
    h_x1 = sum(hx)
    h_x2 = sum(hy)

            # Information calculations
    I_x1_x2_kde = (h_x1 + h_x2 - h_x1_x2)

    return I_x1_x2_kde


def lencorr1(x, y) :  # y=len(y) #k=len(tr)
    lc1 = np.empty ((y, 1))
    lc1[:] = np.NaN
    lc1[:len (x)] = x
    return lc1


def lencorr2(x, y, k) :
    lc2 = np.empty ((y + k, 1))
    lc2[:] = np.NaN
    lc2[k :] = x
    return lc2

def th_delete(tensor, indices) :
    mask = torch.ones ((1,tensor.numel ()), dtype=torch.bool)
    mask[:,indices] = False
    return tensor[mask]
