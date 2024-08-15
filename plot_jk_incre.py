import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import utils
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import utils_fglstm as utils_fg



# var_list1=['SO4','NO3','DOC','Fe','Al','Ca']
# var=[27,21,20,47,37,41]
# var_listn=['$\mathregular{SO_4^{2-}}$','$\mathregular{NO_3^{-}}$','$\mathregular{DOC}$','$\mathregular{Fe^{2+}}$','$\mathregular{Al^{3+}}$','$\mathregular{Ca^{2+}}$']

var_list1=['Mg','Na','Cl']
var=[36,35,29]
var_listn=['$\mathregular{Mg^{2+}}$','$\mathregular{Na^{+}}$','$\mathregular{Cl^{-}}$']

fr1=np.arange(0.25,0.75,0.012) #.012 1 week


frsuper=[fr1]

no=0  #folder number
week1=['1week','2week','3week','4week','5week','6week']
weekfix=['1week']


wn1 = [1]

dirct1='/home/.../incre_mlstmwog/'
dirct2='/home/.../incre_reglstmwog/'


dirct=[dirct1,dirct2]

method = ['LSTM_std', 'LSTM_fg']

frac=[0,int((.35-.25)/.012),int((.45-.25)/.012),int((.55-.25)/.012),int((.65-.25)/.012)]

bin=1000

mse=np.ones((len(var_list1),len(wn1),len(fr1),len(dirct)))
Qsd=np.ones((len(var_list1),len(wn1),len(fr1),len(dirct)))

u11,l11=0,1100    #NSE range for entire data (75% have been used)
ns_sort1=np.ones((len(var_list1),len(wn1),len(dirct),l11-u11))
cdf1=np.ones((len(var_list1),len(wn1),len(dirct),l11-u11))

gmse=np.ones((len(var_list1),len(wn1),len(dirct)))
gmsetest=np.ones((len(var_list1),len(wn1),len(dirct)))
gprediction=np.ones((len(var_list1),len(dirct),1300))
gobserved=np.ones((len(var_list1),len(dirct),1300))
gprediction[:]=np.NaN
gobserved[:]=np.NaN

training_set = pd.read_excel ('//home/.../sd01.xlsx', sheet_name="7hour edited data", header=0)
Q = training_set.iloc[3646 :-1, [68]].values
gi, gf = 504, 2200
Q = Q[gi:gf]
Q = utils_fg.intp (Q)
Q=Q[:1694]  #len(total data poitns)


for dr in range(len(dirct)):
    for vr in range(len(var_list1)):
        var_list = var_list1[vr]
        print('variable:'+str(vr))
        for w in range(len(wn1)):
            wn=wn1[w]
            week = week1[wn - 1]
            weeknum = [20 * wn]
            for fre in range(len(frsuper)):
                fr=frsuper[fre]
                weeknum_ind=fre
                tau=20

                training_set = pd.read_excel ('//home/.../sd01.xlsx', sheet_name="7hour edited data",header=0)

                training_set1 = training_set.iloc[3646 :-1, [var[vr]]].values

                gi, gf = 504, 2200
                X = training_set1[gi :gf]

                pr = pd.read_csv(dirct[dr]+var_list+'_tau_1_'+weekfix[weeknum_ind]+'_ fr_prediction_MSEloss_hiddensize_tau.csv')
                obs = pd.read_csv(dirct[dr]+var_list+'_tau_1_'+weekfix[weeknum_ind]+'_ fr_observed_MSEloss_hiddensize_tau.csv')
                mse_tr = pd.read_csv(dirct[dr]+var_list+'_tau_1_'+weekfix[weeknum_ind]+'_ fr_msetrain_MSEloss_hiddensize_tau.csv')
                mse_ts = pd.read_csv(dirct[dr]+var_list+'_tau_1_'+weekfix[weeknum_ind]+'_ fr_msetest_MSEloss_hiddensize_tau.csv')

                sc = MinMaxScaler ()
                X_sc = sc.fit_transform (X)

                pr11=pr.iloc[:,1:].values
                obs11=obs.iloc[:,1:].values

                bpr,bobs=np.empty((len(pr11),1)),np.empty((len(pr11),1))

                mse_tr,mse_ts=mse_tr.iloc[:,1:].values, mse_ts.iloc[:,1:].values #r2_tr.iloc[:,1:].values, r2_ts.iloc[:,1:].values
                for i in range(len(fr)):
                    tp, lp=np.min (np.min(mse_tr[:,i])), np.where (mse_tr[:,i] == np.min (np.min(mse_tr[:,i])))
                    fp=len(fr)
                    hp=lp[0][0]
                    bpr=np.hstack((bpr,pr11[:,(fp)*(hp)+i].reshape((len(pr11),1))))
                    bobs=np.hstack((bobs,obs11[:,(fp)*(hp)+i].reshape((len(pr11),1))))


                bpr,bobs=bpr[:,1:],bobs[:,1:]


                n=50 #offset
                pr_ts = bpr[n :int (len (bpr) * fr[0]), 0] # ,npr11[n:train_size],nobs11[n:train_size]
                pr_ts1=pr_ts.reshape((len(pr_ts),1))
                obs_ts = bobs[n :int (len (obs) * fr[0]), 0] # ,npr11[n:train_size],nobs11[n:train_size]
                obs_ts1=obs_ts.reshape((len(obs_ts),1))
                for nn in range(int(len(fr)/wn)):
                    train_size = int (len (bpr) * fr[nn*wn])
                    test_size = len (bpr) - train_size


                    pr_1=bpr[train_size:train_size+weeknum[weeknum_ind],nn*wn]
                    obs_1=bobs[train_size:train_size+weeknum[weeknum_ind],nn*wn]
                    mse[vr, w, nn * wn, dr] = sqrt (mean_squared_error (obs_1, pr_1))
                    Qsd[vr, w, nn * wn, dr] = np.std (Q[train_size:train_size+weeknum[weeknum_ind]])
                    pr_ts1=np.append(pr_ts1,(pr_1).reshape((weeknum[weeknum_ind],1)),axis=0)      #,npr11[train_size:],nobs11[train_size:]
                    obs_ts1 = np.append (obs_ts1, (obs_1).reshape ((weeknum[weeknum_ind], 1)),axis=0)  # ,npr11[train_size:],nobs11[train_size:]

                pr1=pr_ts1
                obs1=obs_ts1


                prediction=pr1.reshape((len(pr1),1))
                observed=obs1.reshape((len(obs1),1))


                prediction=sc.inverse_transform(prediction)
                observed=sc.inverse_transform(observed)
                gprediction[vr,dr,:len(prediction)]=prediction.reshape ((len(prediction),))
                gobserved[vr,dr,:len(prediction)]=observed.reshape ((len(prediction),))
                meano=np.mean(observed)
                gmse[vr, w, dr] = sqrt (mean_squared_error (observed, prediction))
                pretest=prediction[400:]
                obtest=observed[400:]
                gmsetest[vr, w, dr] = sqrt (mean_squared_error (obtest, pretest))


                ns1 = utils_fg.nse (prediction[u11:l11], observed[u11:l11])
                ns=ns1.reshape((1,l11-u11))
                ns_sort=np.sort(ns)
                cdf = np.arange (1, len (ns1) + 1) / len (ns1)

                ns_sort1[vr, w, dr, :]=ns_sort
                cdf1[vr, w, dr, :]=cdf


########################  ALL IN ONE PLOT (RMSE, NSE, Q-Q)###########################

################################### for all the variables ############################################

ns1=np.arange(0,1,1/bin)
dirc='/home/.../plot_folder/'
msetest=np.empty((len(var_list1),2))
variable=var_listn

for k in range(len(variable)):
    fig, ax1 = plt.subplots ()
    u,l=0,1212
    x=np.linspace(0,500,100)
    ax1.plot(x,x,':k')

    sc1=ax1.scatter(gobserved[k,1,u:l],gprediction[k,1,u:l],s=20,color='darkorange',facecolors='none')
    sc2=ax1.scatter (gobserved[k,0,u:l], gprediction[k, 0, u :l], s=20,color='#1f77b4',facecolors='none')
    if variable[k]=='$\mathregular{Al^{3+}}$' or variable[k]=='$\mathregular{Fe^{2+}}$':
        ax1.set_xlabel('Observed ug/l')
        ax1.set_ylabel('Predicted ug/l')
    else:
        ax1.set_xlabel ('Observed mg/l')
        ax1.set_ylabel ('Predicted mg/l')
    ax1.legend([sc1,sc2],['regLSTM','mLSTM'],loc='center left')
    ax1.set_title('Observed Vs Predicted '+variable[k])


    b11, b1, h1, w1 = .21, .55, .21, .21
    ax3 = fig.add_axes ([b11, b1, w1, h1])
    for i in range(len(dirct)):
        ax3.plot(ns_sort1[k, 0, i, :],cdf1[k, 0, i, :])
    ax3.legend(['LSTM_fg','LSTM_std'],fontsize=6)
    ax3.set_xlim ([0, 1])
    ax3.set_xlabel ("NSE", fontsize=10)
    ax3.set_ylabel ("CDF", fontsize=10)

    u1, l1 = u,l    #200, 1000
    coef = np.polyfit (np.squeeze (gobserved[k,1,u1:l1]), np.squeeze (gprediction[k,1,u1:l1]), 1)
    poly1d_fn = np.poly1d (coef)
    coefficient_of_dermination = r2_score (gobserved[k,1,u1:l1], gprediction[k,1,u1:l1])

    coef1 = np.polyfit (np.squeeze (gobserved[k,0,u1:l1]), np.squeeze (gprediction[k,0,u1:l1]), 1)
    poly1d_fn1 = np.poly1d (coef1)
    coefficient_of_dermination1 = r2_score (gobserved[k,0,u1:l1], gprediction[k,0,u1:l1])

    linear_regressor,linear_regressor1 = LinearRegression (),LinearRegression ()  # create object for the class
    linear_regressor.fit (gobserved[k,1,u1:l1].reshape(-1, 1),gprediction[k,1,u1:l1].reshape(-1, 1))  # perform linear regression
    linear_regressor1.fit (gobserved[k, 0, u1:l1].reshape(-1, 1), gprediction[k, 0, u1 :l1].reshape(-1, 1))  # perform linear regression
    Y_pred = linear_regressor.predict (gobserved[k,1,u1:l1].reshape(-1, 1))  # make predictions
    Y_pred1 = linear_regressor1.predict (gobserved[k, 0, u1 :l1].reshape(-1, 1))  # make predictions

    ax1.plot(gobserved[k,1,u1:l1],Y_pred,'--',color='darkorange')
    ax1.plot (gobserved[k, 0, u1 :l1], Y_pred1,'--',color='#1f77b4')

    uplim=max(np.nanmax(gobserved[k,0,u:l]),np.nanmax(gprediction[k,0,u:l]),np.nanmax(gobserved[k,1,u:l]),np.nanmax(gprediction[k,1,u:l]))
    lowlim=min(np.nanmin(gobserved[k,0,u:l]),np.nanmin(gprediction[k,0,u:l]),np.nanmin(gobserved[k,1,u:l]),np.nanmin(gprediction[k,1,u:l]))
    ax1.set_xlim([lowlim,uplim])
    ax1.set_ylim([lowlim,uplim])


    ax1.legend ([sc2, sc1], ['LSTM_fg, $r^2$='+ str (round (coefficient_of_dermination1, 3)),'LSTM_std, $r^2$='+ str (round (coefficient_of_dermination, 3))], loc='upper center')
    # plt.savefig (dirc+var_list1[k]+' allinone_plot_incri'+'.png')
    plt.show()

qnan=np.empty((len(fr1)))
qnan[:]=np.nan
for k in range (len (variable)) :
    fig, ax2 = plt.subplots ()
    ax20 = ax2.twinx ()
    for i in range(len(dirct)):
        ax2.scatter (fr1, utils.smooth (mse[k, 0, :, i], 5), s=6)
        ax20.scatter (fr1, utils.smooth (Qsd[k, 0, :, i], 5),s=6,color='#929591')

    ax20.set_ylabel ("Q_sd ($m^3/s$)", fontsize=10)
    ax20.set_ylim (0, 1.5)
    ax2.scatter(fr1,qnan,color='#929591',s=6)
    ax2.legend(['LSTM_fg','LSTM_std','Q_sd'],fontsize=6)
    ax2.set_xlabel ("Week", fontsize=10)
    ax2.set_xticks ([.25, .5, .74], ['20', '40', '60'],rotation=0,fontsize=8.5)  # Set text labels and properties
    if variable[k] == '$\mathregular{Al^{3+}}$' or variable[k] == '$\mathregular{Fe^{2+}}$' :
        ax2.set_ylabel ("RMSE (ug/l)", fontsize=10)
    else:
        ax2.set_ylabel ("RMSE (mg/l)", fontsize=10)
    ax2.set_ylim (0, .2)
    ax2.set_title ('Weekly $RMSE$ '+variable[k], fontsize=10)
    # plt.savefig (dirc+var_list1[k]+' rmse_incri'+'.png')
    plt.show()




############################  G RMSE plots for week 1 to week 6   #################################################

wn1 = [1,2,3,4,5,6]

for dr in range (len (dirct)) :
    for vr in range (len (var_list1)) :
        var_list = var_list1[vr]
        print ('variable:' + str (vr))
        for w in range (len (wn1)) :
            wn = wn1[w]
            week = week1[wn - 1]
            weeknum = [20 * wn]
            for fre in range (len (frsuper)) :
                fr = frsuper[fre]
                weeknum_ind = fre
                tau = 20

                training_set = pd.read_excel ('//home/.../sd01.xlsx',
                                              sheet_name="7hour edited data", header=0)

                training_set1 = training_set.iloc[3646 :-1, [var[vr]]].values

                gi, gf = 504, 2200
                X = training_set1[gi :gf]

                pr = pd.read_csv (dirct[dr] + var_list + '_tau_1_' + weekfix[weeknum_ind] + '_ fr_prediction_MSEloss_hiddensize_tau.csv')
                obs = pd.read_csv (dirct[dr] + var_list + '_tau_1_' + weekfix[weeknum_ind] + '_ fr_observed_MSEloss_hiddensize_tau.csv')
                mse_tr = pd.read_csv (dirct[dr] + var_list + '_tau_1_' + weekfix[weeknum_ind] + '_ fr_msetrain_MSEloss_hiddensize_tau.csv')
                mse_ts = pd.read_csv (dirct[dr] + var_list + '_tau_1_' + weekfix[weeknum_ind] + '_ fr_msetest_MSEloss_hiddensize_tau.csv')

                sc = MinMaxScaler ()
                X_sc = sc.fit_transform (X)

                pr11 = pr.iloc[:, 1 :].values
                obs11 = obs.iloc[:, 1 :].values

                bpr, bobs = np.empty ((len (pr11), 1)), np.empty ((len (pr11), 1))

                mse_tr, mse_ts = mse_tr.iloc[:, 1 :].values, mse_ts.iloc[:,1 :].values  # r2_tr.iloc[:,1:].values, r2_ts.iloc[:,1:].values
                for i in range (len (fr)) :
                    tp, lp = np.min (np.min (mse_tr[:, i])), np.where (mse_tr[:, i] == np.min (np.min (mse_tr[:, i])))
                    fp = len (fr)
                    hp = lp[0][0]
                    bpr = np.hstack ((bpr, pr11[:, (fp) * (hp) + i].reshape ((len (pr11), 1))))
                    bobs = np.hstack ((bobs, obs11[:, (fp) * (hp) + i].reshape ((len (pr11), 1))))

                bpr, bobs = bpr[:, 1 :], bobs[:, 1 :]

                n = 50  # offset
                pr_ts = bpr[n :int (len (bpr) * fr[0]), 0]  # ,npr11[n:train_size],nobs11[n:train_size]
                pr_ts1 = pr_ts.reshape ((len (pr_ts), 1))
                obs_ts = bobs[n :int (len (obs) * fr[0]), 0]  # ,npr11[n:train_size],nobs11[n:train_size]
                obs_ts1 = obs_ts.reshape ((len (obs_ts), 1))
                for nn in range (int (len (fr) / wn)) :
                    train_size = int (len (bpr) * fr[nn * wn])
                    test_size = len (bpr) - train_size

                    pr_1 = bpr[train_size :train_size + weeknum[weeknum_ind], nn * wn]
                    obs_1 = bobs[train_size :train_size + weeknum[weeknum_ind], nn * wn]
                    mse[vr, w, nn * wn, dr] = sqrt (mean_squared_error (obs_1, pr_1))
                    Qsd[vr, w, nn * wn, dr] = np.std (Q[train_size :train_size + weeknum[weeknum_ind]])
                    pr_ts1 = np.append (pr_ts1, (pr_1).reshape ((weeknum[weeknum_ind], 1)),axis=0)  # ,npr11[train_size:],nobs11[train_size:]
                    obs_ts1 = np.append (obs_ts1, (obs_1).reshape ((weeknum[weeknum_ind], 1)),axis=0)  # ,npr11[train_size:],nobs11[train_size:]

                pr1 = pr_ts1
                obs1 = obs_ts1

                prediction = pr1.reshape ((len (pr1), 1))
                observed = obs1.reshape ((len (obs1), 1))

                prediction = sc.inverse_transform (prediction)
                observed = sc.inverse_transform (observed)
                gprediction[vr, dr, :len (prediction)] = prediction.reshape ((len (prediction),))
                gobserved[vr, dr, :len (prediction)] = observed.reshape ((len (prediction),))
                meano = np.mean (observed)
                gmse[vr, w, dr] = sqrt (mean_squared_error (observed, prediction))
                pretest = prediction[400 :]
                obtest = observed[400 :]
                gmsetest[vr, w, dr] = sqrt (mean_squared_error (obtest, pretest))

                ns1 = utils_fg.nse (prediction[u11 :l11], observed[u11 :l11])
                ns = ns1.reshape ((1, l11 - u11))
                ns_sort = np.sort (ns)
                cdf = np.arange (1, len (ns1) + 1) / len (ns1)

                ns_sort1[vr, w, dr, :] = ns_sort
                cdf1[vr, w, dr, :] = cdf

mse[mse==1]=np.NaN

plt.figure()
for k in range(len(variable)):
    for i in range (len (dirct)) :
            plt.plot(wn1,gmse[k,:,i],linewidth=4)
    plt.legend(['LSTM_fg','LSTM_std'], fontsize=12)

    plt.xlabel ('Week',fontsize=12)

    if var_list1[k]=='Al' or var_list1[k]=='Fe':
        plt.ylabel('RMSE ug/l',fontsize=12)
    else:
        plt.ylabel ('RMSE mg/l',fontsize=12)

    plt.title('RMSE '+variable[k],fontsize=12)

    # plt.savefig (dirc+var_list1[k]+' Grmse_incri'+'.png')
    plt.show()


####################### 1 week prediction vs observed ############################
fr1=np.arange(0.25,0.75,0.012) #.012 1 week

frsuper=[fr1]

no=0  #folder number
week=['1week']
weeknum=[20]  #20 data points in 1 week
wk=1


dirct1 = '/home/.../incre_mlstmwog/'
dirct2 = '/home/.../incre_reglstmwog/'
#
dirct=[dirct1,dirct2]

method = ['LSTM_fg', 'LSTM_std']


training_set = pd.read_excel ('//home/.../sd01.xlsx', sheet_name="7hour edited data",header=0)
training_set1 = training_set.iloc[3646 :-1, [68]].values
Q=training_set1[gi:gf]


gprediction=np.ones((len(var_list1),len(frsuper),len(dirct),1213))
gobserved=np.ones((len(var_list1),len(frsuper),len(dirct),1213))
gprediction[:]=np.NaN
gobserved[:]=np.NaN
zpr_tr1, zobs_tr1=np.ones((len(var_list1),len(frsuper),len(dirct),650)),np.ones((len(var_list1),len(frsuper),len(dirct),650))
zpr_ts1, zobs_ts1=np.ones((len(var_list1),len(frsuper),len(dirct),650)),np.ones((len(var_list1),len(frsuper),len(dirct),650))
zpr_tr1[:], zobs_tr1[:],zpr_ts1[:], zobs_ts1[:]=np.NaN,np.NaN,np.NaN,np.NaN

for vr in range(len(var_list1)):
    var_list = var_list1[vr]
    print('variable:'+str(vr))
    for dr in range(len(dirct)):
        for fre in range(len(frsuper)):
            fr=frsuper[fre]
            weeknum_ind=fre

            training_set = pd.read_excel ('//home/.../sd01.xlsx', sheet_name="7hour edited data",header=0)

            training_set1 = training_set.iloc[3646 :-1, [var[vr]]].values

            gi, gf = 504, 2200
            X = training_set1[gi :gf]
            # for fo in range(len(folder)):
            pr = pd.read_csv(dirct[dr]+var_list+'_tau_1_'+week[weeknum_ind]+'_ fr_prediction_MSEloss_hiddensize_tau.csv')
            obs = pd.read_csv(dirct[dr]+var_list+'_tau_1_'+week[weeknum_ind]+'_ fr_observed_MSEloss_hiddensize_tau.csv')
            mse_tr = pd.read_csv(dirct[dr]+var_list+'_tau_1_'+week[weeknum_ind]+'_ fr_msetrain_MSEloss_hiddensize_tau.csv')
            mse_ts = pd.read_csv(dirct[dr]+var_list+'_tau_1_'+week[weeknum_ind]+'_ fr_msetest_MSEloss_hiddensize_tau.csv')

            sc = MinMaxScaler ()
            X_sc = sc.fit_transform (X)

            pr11=pr.iloc[:,1:].values
            obs11=obs.iloc[:,1:].values

            bpr,bobs=np.empty((len(pr11),1)),np.empty((len(pr11),1))

            mse_tr,mse_ts=mse_tr.iloc[:,1:].values, mse_ts.iloc[:,1:].values #r2_tr.iloc[:,1:].values, r2_ts.iloc[:,1:].values
            for i in range(len(fr)):

                tp, lp=np.min (np.min(mse_tr[:,i])), np.where (mse_tr[:,i] == np.min (np.min(mse_tr[:,i])))
                fp=len(fr)
                hp=lp[0][0]
                bpr=np.hstack((bpr,pr11[:,(fp)*(hp)+i].reshape((len(pr11),1))))
                bobs=np.hstack((bobs,obs11[:,(fp)*(hp)+i].reshape((len(pr11),1))))


            bpr,bobs=bpr[:,1:],bobs[:,1:]


            n=50 #offset
            ofr=0
            pr_ts = bpr[n :int (len (bpr) * fr[ofr]), ofr] # ,npr11[n:train_size],nobs11[n:train_size]
            pr_ts1=pr_ts.reshape((len(pr_ts),1))
            obs_ts = bobs[n :int (len (obs) * fr[ofr]), ofr] # ,npr11[n:train_size],nobs11[n:train_size]
            obs_ts1=obs_ts.reshape((len(obs_ts),1))
            for nn in range(int(len(fr)/wk)):
                train_size = int (len (bpr) * fr[nn*wk])
                test_size = len (bpr) - train_size

                pr_1=bpr[train_size:train_size+weeknum[weeknum_ind],nn*wk]
                obs_1=bobs[train_size:train_size+weeknum[weeknum_ind],nn*wk]
                pr_ts1=np.append(pr_ts1,(pr_1).reshape((weeknum[weeknum_ind],1)),axis=0)      #,npr11[train_size:],nobs11[train_size:]
                obs_ts1 = np.append (obs_ts1, (obs_1).reshape ((weeknum[weeknum_ind], 1)),axis=0)  # ,npr11[train_size:],nobs11[train_size:]

            pr1=pr_ts1
            obs1=obs_ts1


            prediction=pr1.reshape((len(pr1),1))
            observed=obs1.reshape((len(obs1),1))


            prediction=sc.inverse_transform(prediction)
            observed=sc.inverse_transform(observed)

            gprediction[vr,0,dr,:len(prediction)]=prediction.reshape((len(prediction),))
            gobserved[vr,0,dr,:len(observed)]=observed.reshape((len(observed),))
            # 200 points around 50% data
            windowsize=200
            train_size1 = int (len (bpr) * fr[0])
            prediction1=prediction[train_size1:train_size1+windowsize]
            observed1=observed[train_size1:train_size1+windowsize]

            zpr_ts1[vr,0,dr,:windowsize],zobs_ts1[vr,0,dr,:windowsize]=prediction1.reshape((len(prediction1),)),observed1.reshape((len(observed1),))

            difl = len (Q) - len (observed)
            Q1 = Q[n :len(observed)+n]


training_set = pd.read_excel ('//home/.../sd01.xlsx', sheet_name="7hour edited data", header=0)
Q = training_set.iloc[3646 :-1, [68]].values
gi, gf = 504, 2200
Q = Q[gi:gf]
Q = utils_fg.intp (Q)
Q = Q[50:]
Q[int(len(Q)*.75):] = np.nan
Q1=np.empty((1694,1))
Q1[:]=np.nan
Q1[:1646]=Q
Q=Q1

dircts='/home/.../plots_folder/'

for vr in range(len(var_list1)):
    fig, ax1 = plt.subplots (2)
    ax1[0].plot (gobserved[vr,0,0,:],color='black',linewidth=1)
    ax1[0].plot (gprediction[vr,0,0,:],color='#1f77b4',linewidth=1.5)

    # ax1[0].set_xlabel ('time steps')

    ax1[1].plot (gobserved[vr,0,1,:],color='black',linewidth=1)
    ax1[1].plot (gprediction[vr,0,1,:],color='darkorange',linewidth=1.5)

    ax1[1].set_xlabel ('time steps')

    if var_list1[vr] == 'Al' or var_list1[vr] == 'Fe' :
        ax1[0].set_ylabel (var_listn[vr] + '  ug/l')
        ax1[1].set_ylabel (var_listn[vr] + '  ug/l')
    else :
        ax1[0].set_ylabel (var_listn[vr] + '  mg/l')
        ax1[1].set_ylabel (var_listn[vr] + '  mg/l')


    ax20=ax1[0].twinx()
    ax21=ax1[1].twinx()
    ax20.plot(Q,color='#929591')
    t = np.arange (0, int(len(pr11)), 1)
    ax20.fill_between(t, 0,np.squeeze(Q),color='#929591', alpha=0.7)
    ax21.plot(Q,color='#929591')
    ax21.fill_between(t,0,np.squeeze(Q),color='#929591', alpha=0.7)

    ax20.set_ylim (0, 3)
    ax21.set_ylim (0, 3)
    ax20.set_ylabel ('Q $m^3/s$')
    ax21.set_ylabel ('Q $m^3/s$')
    ax1[0].plot (np.nan,color='#929591')
    ax1[1].plot (np.nan, color='#929591')

    ax1[0].axvline (x=len (pr_ts), ymin=0, color='red', linestyle='dotted', linewidth=2)
    ax1[1].axvline (x=len (pr_ts), ymin=0, color='red', linestyle='dotted', linewidth=2)

    fig.suptitle(var_listn[vr] +' '+week[0]+' incremental predictions')
    ax1[0].set_title (method[0])  # +' Window='+str(win))
    ax1[0].legend (['Observed', 'Predicted','Q'], fontsize=10, loc='upper right')
    ax1[1].set_title (method[1])  # +' Window='+str(win))
    ax1[1].legend (['Observed', 'Predicted','Q'], fontsize=10, loc='upper right')

    ax1[0].xaxis.set_ticks ([0, len (pr11) / 4, 2 * len (pr11) / 4, 3 * len (pr11) / 4, 4 * len (pr11) / 4])
    ax1[1].xaxis.set_ticklabels (['0', '20 week', '40 week', '60 week', '80 week'])
    ax1[1].xaxis.set_ticks ([0, len (pr11) / 4, 2 * len (pr11) / 4, 3 * len (pr11) / 4, 4 * len (pr11) / 4])
    ax1[0].xaxis.set_ticklabels ([])

    # plt.savefig (dircts+var_list1[vr]+'_obs-pred_incri.png')
    plt.show ()

for vr in range (len (var_list1)) :
    plt.plot(zobs_ts1[vr,0,0,:],color='black',linewidth=1)
    plt.plot(zpr_ts1[vr,0,0,:],color='#1f77b4',linewidth=1.5)
    plt.plot(zpr_ts1[vr,0,1,:],color='darkorange',linewidth=1.5)
    # plt.savefig (dircts+var_list1[vr]+'_zommedin_incri.png')
    plt.show()




