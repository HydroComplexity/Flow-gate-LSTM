import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import utils_fglstm as utils_fg


var=[41,36,35,29,37,47,20,40,27,21]  #new network
var_list=['Ca','Mg','Na','Cl','Al','Fe','DOC','K','SO4','NO3'] #new network
invar_list1=[['Q','Ca'],['Q','Ca','Mg'],['Q','Ca','Cl','Mg','Na'],['Q','Mg','Na','Cl'],
             ['Q','Fe','DOC','Al'],['Q','Al','DOC','Cl','Fe'],['Q','Fe','DOC'],['Q','Fe','DOC','K'],
             ['Q','NO3','Ca','SO4'],['Q','SO4','DOC','Cl','NO3']]

var_listn=['$\mathregular{Ca^{2+}}$','$\mathregular{Mg^{2+}}$','$\mathregular{Na^{+}}$','$\mathregular{Cl^{-}}$','$\mathregular{Al^{3+}}$','$\mathregular{Fe^{2+}}$','$\mathregular{DOC}$','$\mathregular{SO_4^{2-}}$','$\mathregular{NO_3^{-}}$']


f1=2 #fration data used for nse 50% of data
f=f1

fr=[0.25,0.37,0.5,0.67,0.75]

windowsize1=200   #zoomed in plot
window=[1]
no=0  #folder number
tau=1
method=['regLSTM','mLSTM(tanh)']
methodid=1
ind=0


dirct1='/home/.../output_mlstm/'
dirct2='/home/.../output_reglstm/'


dirct=[dirct1,dirct2]

bin=1000


mse=np.ones((len(var_list),len(fr),len(dirct)))
tsmse=np.ones((len(var_list),len(fr),len(dirct)))
trmse=np.ones((len(var_list),len(fr),len(dirct)))


u11,v11=0,1500

ns_sort1=np.ones((len(var_list),len(fr),len(dirct),v11-u11))
cdf1=np.ones((len(var_list),len(fr),len(dirct),v11-u11))

gprediction=np.ones((len(var_list),len(fr),len(dirct),1700))
gobserved=np.ones((len(var_list),len(fr),len(dirct),1700))
gprediction[:]=np.NaN
gobserved[:]=np.NaN

for dr in range(len(dirct)):
    for vr in range(len(var_list)):
        training_set = pd.read_excel ('//home/.../sd01.xlsx',sheet_name="7hour edited data", header=0)
        training_set1 = training_set.iloc[3646 :-1, [var[vr]]].values
        gi, gf = 504, 2200
        X = training_set1[gi :gf]
        for fr1 in range(len(fr)):
            pr = pd.read_csv (dirct[dr] + var_list[vr] + '_' + str (invar_list1[vr]) + '_tau_' + str (tau) + '_fr_' + str (fr[fr1]) + '_nprediction_MSEloss_hiddensize_tau.csv')
            obs = pd.read_csv (dirct[dr] + var_list[vr] + '_' + str (invar_list1[vr]) + '_tau_' + str (tau) + '_fr_' + str (fr[fr1]) + '_nobserved_MSEloss_hiddensize_tau.csv')

            sc = MinMaxScaler ()
            X_sc = sc.fit_transform (X)

            pr11=pr.iloc[:,1:].values
            obs11=obs.iloc[:,1:].values
            pr11=sc.inverse_transform(pr11)
            obs11=sc.inverse_transform(obs11)

            n=50
            train_size = int (len (pr11) * fr[fr1])   #250
            test_size = len (pr11) - train_size
            windowsize=test_size    #test_size
            lk=0
            pr_tr, obs_tr = pr11[n :train_size], obs11[n :train_size]  # ,npr11[n:train_size],nobs11[n:train_size]
            pr_ts, obs_ts = pr11[lk+train_size :lk+train_size+windowsize], obs11[lk+train_size :lk+train_size+windowsize]  # ,npr11[train_size:],nobs11[train_size:]
            observed=np.concatenate((obs_tr,obs_ts),axis=0)
            prediction=np.concatenate((pr_tr,pr_ts),axis=0)

            mse[vr,fr1,dr]= sqrt(mean_squared_error(observed, prediction, multioutput='raw_values'))
            tsmse[vr,fr1,dr]=sqrt(mean_squared_error(obs_ts, pr_ts, multioutput='raw_values'))
            trmse[vr,fr1,dr]=sqrt(mean_squared_error(obs_tr, pr_tr, multioutput='raw_values'))

            gprediction[vr, fr1,dr, :len (prediction)] = prediction.reshape ((len (prediction),))
            gobserved[vr, fr1,dr, :len (prediction)] = observed.reshape ((len (prediction),))

            # NSE vs CDF
            ns1 = utils_fg.nse(prediction[u11:v11], observed[u11:v11])
            ns = ns1.reshape ((1, v11 - u11))
            ns_sort = np.sort (ns)
            cdf = np.arange (1, len (ns1) + 1) / len (ns1)
            ns_sort1[vr, fr1, dr, :] = ns_sort
            cdf1[vr, fr1, dr, :] = cdf



ns1=np.arange(0,1,1/bin)
dirc='/home/.../plots_folder/'

variable=var_list #'Fe','DOC','K',

for k in range(len(variable)):
    fig, ax1 = plt.subplots ()
    u,l=0,1643
    x=np.linspace(0,500,100)
    ax1.plot(x,x,':k')

    sc1=ax1.scatter(gobserved[k,f,1,u:l],gprediction[k,f,1,u:l],s=20,color='darkorange',facecolors='none')
    sc2=ax1.scatter (gobserved[k,f,0,u:l], gprediction[k,f, 0, u :l], s=20,color='#1f77b4',facecolors='none')
    if var_list[k] == 'Al' or var_list[k] == 'Fe' :
        ax1.set_xlabel('Observed ug/l')
        ax1.set_ylabel('Predicted ug/l')
    else:
        ax1.set_xlabel('Observed mg/l')
        ax1.set_ylabel('Predicted mg/l')
    ax1.legend([sc1,sc2],['LSTM_std','LSTM_fg'],loc='center left')
    ax1.set_title('Observed Vs Predicted '+var_listn[k]+' fr=0.5')

    b11, b1, h1, w1 = .21, .55, .21, .21
    ax3 = fig.add_axes ([b11, b1, w1, h1])
    for i in range(len(dirct)):
        ax3.plot(ns_sort1[k, f1, i, :],cdf1[k, f1, i, :])
    ax3.legend(['LSTM_fg','LSTM_std'],fontsize=6)
    ax3.set_xlim ([0, 1])
    ax3.set_xlabel ("NSE", fontsize=10)
    ax3.set_ylabel ("CDF", fontsize=10)

    u1, l1 = 0, 1643
    coef = np.polyfit (np.squeeze (gobserved[k,f,1,u1:l1]), np.squeeze (gprediction[k,f,1,u1:l1]), 1)
    poly1d_fn = np.poly1d (coef)
    coefficient_of_dermination = r2_score (gobserved[k,f,1,u1:l1], gprediction[k,f,1,u1:l1])

    coef1 = np.polyfit (np.squeeze (gobserved[k,f,0,u1:l1]), np.squeeze (gprediction[k,f,0,u1:l1]), 1)
    poly1d_fn1 = np.poly1d (coef1)
    coefficient_of_dermination1 = r2_score (gobserved[k,f,0,u1:l1], gprediction[k,f,0,u1:l1])


    linear_regressor,linear_regressor1 = LinearRegression (),LinearRegression () # create object for the class
    linear_regressor.fit (gobserved[k,f,1,u1:l1].reshape(-1, 1),gprediction[k,f,1,u1:l1].reshape(-1, 1))  # perform linear regression
    linear_regressor1.fit (gobserved[k,f, 0, u1:l1].reshape(-1, 1), gprediction[k,f, 0, u1 :l1].reshape(-1, 1))  # perform linear regression


    Y_pred = linear_regressor.predict (gobserved[k,f,1,u1:l1].reshape(-1, 1))  # make predictions
    Y_pred1 = linear_regressor1.predict (gobserved[k,f, 0, u1 :l1].reshape(-1, 1))  # make predictions

    ax1.plot(gobserved[k,f,0,u1:l1],Y_pred,'--',color='darkorange')
    ax1.plot (gobserved[k,f, 1, u1 :l1], Y_pred1,'--',color='#1f77b4')

    uplim=max(np.nanmax(gobserved[k,f,0,u:l]),np.nanmax(gprediction[k,f,0,u:l]),np.nanmax(gobserved[k,f,1,u:l]),np.nanmax(gprediction[k,f,1,u:l]))
    lowlim=min(np.nanmin(gobserved[k,f,0,u:l]),np.nanmin(gprediction[k,f,0,u:l]),np.nanmin(gobserved[k,f,1,u:l]),np.nanmin(gprediction[k,f,1,u:l]))
    ax1.set_xlim([lowlim,uplim])
    ax1.set_ylim([lowlim,uplim])


    ax1.legend ([sc2, sc1], ['LSTM_fg, $r^2$='+ str (round (coefficient_of_dermination1, 3)), 'LSTM_std, $r^2$='+str (round (coefficient_of_dermination, 3))], loc='upper center')

    # plt.savefig (dirc+var_list[k]+' allinone_plot_noincri'+'.png')
    plt.show()

#######################
var_list1=['Ca','Mg','Na','Cl','Al','Fe','DOC','K','SO4','NO3'] #new network
fr=[0.5]

windowsize=200
window=[1]
no=0  #folder number
tau=1
method=['mLSTM(tanh)','regLSTM']

gprediction=np.ones((len(var_list),len(fr),len(dirct),1694))
gobserved=np.ones((len(var_list),len(fr),len(dirct),1694))
gprediction[:]=np.NaN
gobserved[:]=np.NaN
pr_tr1, obs_tr1=np.ones((len(var_list),len(fr),len(dirct),850)),np.ones((len(var_list),len(fr),len(dirct),850))
pr_ts1, obs_ts1=np.ones((len(var_list),len(fr),len(dirct),850)),np.ones((len(var_list),len(fr),len(dirct),850))
pr_tr1[:], obs_tr1[:],pr_ts1[:], obs_ts1[:]=np.NaN,np.NaN,np.NaN,np.NaN

gerror=np.empty((1696,1))
temp=gerror

for vr in range(len(var_list)):
    training_set = pd.read_excel ('//home/.../sd01.xlsx', sheet_name="7hour edited data",header=0)
    training_set1 = training_set.iloc[3646 :-1, [var[vr]]].values

    gi, gf = 504, 2200
    X = training_set1[gi :gf]
    for dr in range (len (dirct)) :
        for fr1 in range(len(fr)):
            pr = pd.read_csv (dirct[dr] + var_list[vr] + '_' + str (invar_list1[vr]) + '_tau_' + str (tau) + '_fr_' + str (fr[fr1]) + '_nprediction_MSEloss_hiddensize_tau.csv')
            obs = pd.read_csv (dirct[dr] + var_list[vr] + '_' + str (invar_list1[vr]) + '_tau_' + str (tau) + '_fr_' + str (fr[fr1]) + '_nobserved_MSEloss_hiddensize_tau.csv')

            sc = MinMaxScaler ()
            X_sc = sc.fit_transform (X)

            pr11=pr.iloc[:,1:].values
            obs11=obs.iloc[:,1:].values
            pr11=sc.inverse_transform(pr11)
            obs11=sc.inverse_transform(obs11)

            n=50
            train_size = int (len (pr11) * fr[fr1])
            test_size = len (pr11) - train_size
            pr_tr, obs_tr = pr11[n :train_size], obs11[n :train_size]  # ,npr11[n:train_size],nobs11[n:train_size]
            pr_ts, obs_ts = pr11[train_size :train_size+test_size], obs11[train_size :train_size+test_size]  # ,npr11[train_size:],nobs11[train_size:]
            observed=np.concatenate((obs_tr,obs_ts),axis=0)
            prediction=np.concatenate((pr_tr,pr_ts),axis=0)
            error=np.abs((pr11[train_size :train_size+test_size]- obs11[train_size :train_size+test_size]).reshape((test_size,1)))
            temp[:]=np.NaN
            temp[train_size:train_size+test_size]=error
            gerror=np.hstack((gerror,temp))
            difl=len(Q)-len(observed)
            Q1=Q[difl:]
            pr_tr_len=len(pr_tr)
            gobserved[vr,fr1,dr,:len(observed)]=observed.reshape((len(observed),))
            gprediction[vr,fr1,dr,:len(prediction)]=prediction.reshape((len(observed),))

            offset=250
            pr_tr1[vr,fr1,dr,:len(pr_tr)], obs_tr1[vr,fr1,dr,:len(pr_tr)] = pr11[n :train_size].reshape((len(pr_tr),)), obs11[n :train_size].reshape((len(pr_tr),))  # ,npr11[n:train_size],nobs11[n:train_size]
            pr_ts1[vr,fr1,dr,:windowsize], obs_ts1[vr,fr1,dr,:windowsize] = pr11[offset+train_size :offset+train_size + windowsize].reshape((windowsize,)), obs11[offset+train_size :offset+train_size + windowsize].reshape((windowsize,))  # ,npr11[train_size:],nobs11[train_size:]


training_set = pd.read_excel ('//home/.../sd01.xlsx', sheet_name="7hour edited data", header=0)
Q = training_set.iloc[3646 :-1, [68]].values
gi, gf = 504, 2200
Q = Q[gi:gf]
Q = utils_fg.intp (Q)
Q = Q[50:]

dircts='/home/.../plots_folder/'
fig, ax1 = plt.subplots (2)
for vr in range(len(var_list1)):
    ax1[0].plot (gobserved[vr,0,0,:],color='black',linewidth=1)
    ax1[0].plot (gprediction[vr,0,0,:],color='#1f77b4',linewidth=1.5)
    ax1[1].plot (gobserved[vr,0,1,:],color='black',linewidth=1)
    ax1[1].plot (gprediction[vr,0,1,:],color='darkorange',linewidth=1.5)
    ax1[1].set_xlabel ('time steps')

    if var_list[vr]=='Al' or var_list[vr]=='Fe':
        ax1[0].set_ylabel (var_list1[vr] + '  ug/l')
        ax1[1].set_ylabel (var_list1[vr] + '  ug/l')
    else:
        ax1[0].set_ylabel (var_list1[vr] + '  mg/l')
        ax1[1].set_ylabel (var_list1[vr] + '  mg/l')

    ax1[1].xaxis.set_ticks([0,len(gprediction[vr,0,0,:])/4,2*len(gprediction[vr,0,0,:])/4,3*len(gprediction[vr,0,0,:])/4,4*len(gprediction[vr,0,0,:])/4])
    ax1[1].xaxis.set_ticklabels(['0', '20 week', '40 week','60 week','80 week'])
    ax1[0].xaxis.set_ticks([0,len(gprediction[vr,0,0,:])/4,2*len(gprediction[vr,0,0,:])/4,3*len(gprediction[vr,0,0,:])/4,4*len(gprediction[vr,0,0,:])/4])
    ax1[0].set_xticklabels([])

    ax20=ax1[0].twinx()
    ax21=ax1[1].twinx()
    ax20.plot(Q,color='#929591')
    t = np.arange (0, len(Q), 1)
    ax20.fill_between(t, 0,np.squeeze(Q),color='#929591', alpha=0.7)
    ax21.plot(Q,color='#929591')
    ax21.fill_between(t,0,np.squeeze(Q),color='#929591', alpha=0.7)

    ax20.set_ylim (0, 3)
    ax21.set_ylim (0, 3)
    ax20.set_ylabel ('Q $m^3/s$')
    ax21.set_ylabel ('Q $m^3/s$')
    ax1[0].plot (np.nan,color='#929591')
    ax1[1].plot (np.nan, color='#929591')

    ax1[0].axvline (x=pr_tr_len, ymin=0, color='red', linestyle='dotted', linewidth=2)
    ax1[1].axvline (x=pr_tr_len, ymin=0, color='red', linestyle='dotted', linewidth=2)

    fig.suptitle ('Prediction of '+var_list1[vr] +' at fr=' + str (round(fr[0],2)))
    ax1[0].set_title ('LSTM_fg' )#+' Window='+str(win))

    ax1[0].legend (['Observed', 'Predicted','Q'], fontsize=10, loc='upper left')

    ax1[1].set_title ('LSTM_std')#+' Window='+str(win))
    ax1[1].legend (['Observed', 'Predicted','Q'], fontsize=10, loc='upper left')

    # plt.savefig (dircts+var_list[vr]+'_obs-pred_noincri_3lstm.png')
    plt.show ()


plt.figure()
for vr in range (len (var_list)) :
    plt.plot(obs_ts1[vr,0,0,:],color='black',linewidth=1)
    plt.plot(pr_ts1[vr,0,0,:],color='#1f77b4',linewidth=1.5)
    plt.plot(pr_ts1[vr,0,1,:],color='darkorange',linewidth=1.5)
    # plt.savefig (dircts+var_list[vr]+'_zommedin_noincri_3lstm.png')
    plt.show()


