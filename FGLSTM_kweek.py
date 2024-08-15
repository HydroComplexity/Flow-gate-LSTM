import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.optim as optim
import torch
from torch import nn
from sklearn.metrics import mean_squared_error as MSE
from torch.multiprocessing import Process
import os
from lstm_classes import Sequence
from utils import lencorr2, lencorr1
import sys
from tqdm import tqdm
import utils_fglstm as utils_fg # utils_fglstm.py


device = sys.argv[1]
if not torch.cuda.is_available():
    device = 'cpu'

if device=='cuda':
    print("Running of GPU: "+device)
else:
    print("Running on CPU: "+device)



def allvar(training_set,training_set1,var_list,tau,H,num_epochs,fr,inputvar,invar_list,learning_rate,header,seq_length,num_classes,device,grad,mgate,cgate,fluxgate,week,fold):
    #training set and training set1 are the disctionary contains target+input var and target var list

    print('fraction='+str(fr))
    process_id=os.getpid()
    print('######################################')
    print('Process ID'+str(process_id))
    print('######################################')
    print(mgate)

    gmse_train,gmse_test,gprediction1,gobserved1={},{},{},{}
    for i in range(len(var_list)):
        gmse_train[var_list[i]] = np.ones ((1,len (fr)))             #observed testing set: for all hidden_size (number of tau in column)
        gmse_test[var_list[i]] = np.ones ((1,len (fr)))        #observed training set: for all hidden_size (number of tau in column)
        gprediction1[var_list[i]] = np.ones((len(training_set1[var_list[i]])-2,1))
        gobserved1[var_list[i]] = np.ones((len(training_set1[var_list[i]])-2,1))


    training_seti={}
    for j in range (len (var_list)) :
        training_seti[var_list[j]]=training_set[var_list[j]][1:,:]
        flux=training_set[var_list[j]][:,len(inputvar[j])]*training_set[var_list[j]][:,1]  #tar var is at the position len(iputvar)
        slp=utils_fg.getgrad(training_set[var_list[j]][:,1])   #Q is at the position 1
        slp1=utils_fg.getgrad(training_set[var_list[j]][:,len(inputvar[j])] )  #tar var is at the position len(iputvar) #cgate gradient
        slp2=utils_fg.getgrad(flux)   #flux gradient
        training_seti[var_list[j]]=np.hstack((training_seti[var_list[j]],slp.reshape((len(slp),1)),slp1.reshape((len(slp1),1)),slp2.reshape((len(slp2),1))))

    training_set=training_seti

    sc,training_data,x,y={},{},{},{}
    for i in range(len(var_list)):
        sc[var_list[i]] = MinMaxScaler ()
        training_data[var_list[i]] = sc[var_list[i]].fit_transform (training_set[var_list[i]])  # normalisation

        x[var_list[i]], y[var_list[i]] = utils_fg.sliding_windows (training_data[var_list[i]], seq_length)

    nk = tau
    input_size1,diffvar1={},{}
    for i in range(len(var_list)):
        input_size1[var_list[i]] = len (inputvar[i]) * (nk)+1  # number of inputs nk+1
        diffvar1[var_list[i]] = len (inputvar[i]) * (nk)+1   #nk+1

    input_size={}
    for i in range (len (var_list)) :
        if grad == 'no' :
            input_size[var_list[i]] = len (inputvar[i]) * (nk)  # number of inputs
        else :
            input_size[var_list[i]] = len (inputvar[i]) * (nk) + 1  # number of inputs

    for hidden_size in tqdm(H):
        print ('**************************************')
        print('Hidden Size='+str(hidden_size))
        print ('week=' + str (week))
        print ('**************************************')
        gpredt_var,gpredt_train_var,gobs_var,gobs_train_var,mse_mlstmts,mse_mlstmtr={},{},{},{},{},{}
        for i in range(len(var_list)):
            gpredt_var[var_list[i]] = []
            gpredt_train_var[var_list[i]] = []
            gobs_var[var_list[i]] = []
            gobs_train_var[var_list[i]] = []
            mse_mlstmts[var_list[i]] = np.ones (len (fr))    #np.empty ((len (tau), len (H)))
            mse_mlstmtr[var_list[i]] = np.ones (len (fr))
        fk = 0              #initialization of hidden size vector
        for fr1 in fr:
            train_size,test_size,trainX,trainY,testX,testY,input,target,test_target,lss={},{},{},{},{},{},{},{},{},{}
            for i in range (len (var_list)) :
                train_size[var_list[i]] = int(len(y[var_list[i]]) * fr1)
                test_size[var_list[i]] = len(y[var_list[i]]) - train_size[var_list[i]]

                trainX[var_list[i]] = torch.Tensor(np.array(x[var_list[i]][:train_size[var_list[i]],:,:])).to(device)
                trainY[var_list[i]] = torch.Tensor(np.array(y[var_list[i]][:train_size[var_list[i]],:])).to(device)

                testX[var_list[i]] = torch.Tensor(np.array(x[var_list[i]][train_size[var_list[i]]:,:,:])).to(device)
                testY[var_list[i]] = torch.Tensor(np.array(y[var_list[i]][train_size[var_list[i]]:,:])).to(device)

            # if __name__ == '__main__' :
                input[var_list[i]] = trainX[var_list[i]][:,:,input_size1[var_list[i]]+num_classes-diffvar1[var_list[i]]:]                                     #trainX row from 3 and column till last -1
                print (input[var_list[i]].shape)
                target[var_list[i]] = trainY[var_list[i]][:,0:num_classes]                                       #trainY row from 3rd and column from 1 to end
            # test_input = testX[:,:,input_size1+num_classes-diffvar1:]                                    #till row 3 and column till last -1
                test_target[var_list[i]] = testY[var_list[i]][:,0:num_classes]                                  # till row 3 and column from 1 to last
                lss[var_list[i]] = np.ones ((num_epochs, 1))

            if fr1 == fr[0]:
                seq,optimizer={},{}
                criterion = nn.MSELoss().to (device)
                for i in range(len(var_list)):
                    seq[var_list[i]] = Sequence(input_size[var_list[i]], hidden_size, seq_length, num_classes, mgate,cgate,fluxgate,grad, inputvar[i], nk, LSTM=True,custom=True, device=device,stshy=0,).to (device)
                    seq[var_list[i]].float()
                    optimizer[var_list[i]] = optim.Adam (seq[var_list[i]].parameters (), lr=learning_rate)


            parallel_process = []
            manager = torch.multiprocessing.Manager()
            return_dict = manager.dict()
            for k in range(len(var_list)):
                p_am = Process(target=utils_fg.lstmtrain, args=(num_epochs,seq[var_list[k]],optimizer[var_list[k]],criterion,input[var_list[k]],target[var_list[k]],lss[var_list[k]], k, return_dict))
                p_am.start()
                parallel_process.append(p_am)

            for p_am in parallel_process:
                p_am.join()

            for kr in range(len(var_list)):
                seq[var_list[kr]] = return_dict[kr]


            predicQ=training_data[var_list[0]][train_size[var_list[0]]+seq_length+1:,1]     #Q will always be after the target
            predicGQ=training_data[var_list[0]][train_size[var_list[0]]+seq_length+1:,len(invar_list[0])+1]
            testn={}
            for i in range(len(var_list)):
                testn[var_list[i]]=training_data[var_list[i]][train_size[var_list[i]]:train_size[var_list[i]]+seq_length+2,1:]    #removed the 1st target array
            print ('LSTM model is ready to predict')

            pred,x3={},{}
            with torch.no_grad () :  # no gradient calculations
                for o in range(len(var_list)):
                    seq[var_list[o]].eval ()
                    future = 0
                    x2, y2 = utils_fg.sliding_windows (testn[var_list[o]][:, :],seq_length)  # train data is normalized data
                    x3[var_list[o]]=x2[0,:,:]
                    x3[var_list[o]] = x3[var_list[o]].reshape ((1, seq_length, len (inputvar[o]) * (nk)+3))  # (nk+1)
                    testX3 = torch.Tensor (x3[var_list[o]]).to(device)
                    pred1 = seq[var_list[o]](testX3, future=future)
                    pred[var_list[o]] = np.ones ((len (test_target[var_list[0]]), 1))
                    pred[var_list[o]][0] = pred1.detach().cpu().numpy()
                for i in range(len(test_target[var_list[0]])-1):
                    for o in range(len(var_list)):
                        x3[var_list[o]] = np.roll(x3[var_list[o]],-1,axis=1)
                        x3[var_list[o]][0, -1, 0] = predicQ[i]
                        for k in range(len(invar_list[o])-1):  #assuming array Q at the 1st position
                            x3[var_list[o]][0,-1,k+1]=pred[invar_list[o][k+1]][i]

                        x3[var_list[o]][0, -1, len(invar_list[o])] = predicGQ[i]
                        x3[var_list[o]] = x3[var_list[o]].reshape ((1, seq_length, len (inputvar[o]) * (nk)+3))  # (nk+1)
                        testX3 = torch.Tensor (x3[var_list[o]]).to(device)
                        pred1 = seq[var_list[o]](testX3, future=future)
                        pred[var_list[o]][i + 1] = pred1.detach().cpu().numpy()

            pred_train,loss,loss1={},{},{}
            yy,tr,train_trfmp,train_trfmo,test_trfmp,test_trfmo,train_predict1,trainY_plot1,test_predict1,testY_plot1={},{},{},{},{},{},{},{},{},{}
            train_predict, trainY_plot, test_predict, testY_plot,train_ip,train_io,test_ip,test_io ={},{},{},{},{},{},{},{}

            for i in range(len(var_list)):
                pred[var_list[i]]=torch.Tensor(pred[var_list[i]]).to(device)
                pred_train[var_list[i]]=seq[var_list[i]](input[var_list[i]], future=future)
                loss1[var_list[i]]=criterion(pred_train[var_list[i]], target[var_list[i]])
                loss[var_list[i]] = criterion(pred[var_list[i]], test_target[var_list[i]])
                print('using observed values')
                print('train loss:',loss1[var_list[i]].item())
                print ('test loss:', loss[var_list[i]].item ())
                print('Saving the predicted solute')

                yy[var_list[i]] = pred[var_list[i]].detach().cpu().numpy()
                tr[var_list[i]] = pred_train[var_list[i]].detach().cpu().numpy ()
                target[var_list[i]]=target[var_list[i]].cpu()
                test_target[var_list[i]]=test_target[var_list[i]].cpu()
                train_trfmp[var_list[i]]=lencorr1(tr[var_list[i]],len(y[var_list[i]]))
                train_trfmo[var_list[i]]=lencorr1(target[var_list[i]],len(y[var_list[i]]))
                test_trfmp[var_list[i]]=lencorr1(yy[var_list[i]],len(y[var_list[i]]))
                test_trfmo[var_list[i]]=lencorr1(test_target[var_list[i]],len(y[var_list[i]]))

                train_predict1[var_list[i]]=(train_trfmp[var_list[i]])
                trainY_plot1[var_list[i]]=(train_trfmo[var_list[i]])
                test_predict1[var_list[i]] = (test_trfmp[var_list[i]])
                testY_plot1[var_list[i]] = (test_trfmo[var_list[i]])


                train_predict[var_list[i]]=train_predict1[var_list[i]][:len(tr[var_list[i]])]
                trainY_plot[var_list[i]]=trainY_plot1[var_list[i]][:len(target[var_list[i]])]
                test_predict[var_list[i]]=test_predict1[var_list[i]][:len(yy[var_list[i]])]
                testY_plot[var_list[i]]=testY_plot1[var_list[i]][:len(test_target[var_list[i]])]

                if np.isnan (train_predict[var_list[i]]).any () :
                    mse_mlstmtr[var_list[i]][fk] = 11111
                    mse_mlstmts[var_list[i]][fk] = 11111
                    print ('nan in prediction')
                else :
                    mse_mlstmtr[var_list[i]][fk] = MSE(trainY_plot[var_list[i]], train_predict[var_list[i]], multioutput='raw_values')
                    mse_mlstmts[var_list[i]][fk]=MSE(testY_plot[var_list[i]],test_predict[var_list[i]],multioutput='raw_values')


                train_ip[var_list[i]]=lencorr2(train_predict[var_list[i]],train_size[var_list[i]],seq_length)
                train_io[var_list[i]]=lencorr2(trainY_plot[var_list[i]],train_size[var_list[i]],seq_length)
                test_ip[var_list[i]]=lencorr2(test_predict[var_list[i]],test_size[var_list[i]],0)
                test_io[var_list[i]]=lencorr2(testY_plot[var_list[i]],test_size[var_list[i]],0)

                gpredt_var[var_list[i]].append (test_ip[var_list[i]])                      #for every tau
                gpredt_train_var[var_list[i]].append (train_ip[var_list[i]])               #for every tau
                gobs_var[var_list[i]].append (test_io[var_list[i]])                         #for every tau
                gobs_train_var[var_list[i]].append (train_io[var_list[i]])                 #for every tau


            if fk+1<len(fr):
                print('#'*20)
                print("########################### fraction ",str(fr[fk]) , "###########################")
                print("#################", "  ", "Now starting fraction =",str(fr[fk+1]), "#################")
                print(mgate)
                print(week)
                print(hidden_size)
                print("######################################################################################")
            fk=fk+1

        gpredict,gobs,p,o={},{},{},{}
        for i in range(len(var_list)):
            gmse_train[var_list[i]] = np.concatenate ((gmse_train[var_list[i]], mse_mlstmtr[var_list[i]].reshape ((1, len (fr)))))
            gmse_test[var_list[i]] = np.concatenate ((gmse_test[var_list[i]], mse_mlstmts[var_list[i]].reshape ((1, len (fr)))))
            gpredict[var_list[i]] = np.ones ((len (training_set1[var_list[i]])-2, 1))
            gobs[var_list[i]] = np.ones ((len (training_set1[var_list[i]])-2, 1))
        # print(f"{'*'*20}{len(fr)}{'*'*10}{len(gpredt_train_var)}{'*'*10}{len(gobs_train_var)}")
            for j in range (len (fr)) :
                p[var_list[i]] = np.concatenate ((gpredt_train_var[var_list[i]][j], gpredt_var[var_list[i]][j]))
                o[var_list[i]] = np.concatenate ((gobs_train_var[var_list[i]][j], gobs_var[var_list[i]][j]))
                gpredict[var_list[i]] = np.hstack ((gpredict[var_list[i]], p[var_list[i]]))
                gobs[var_list[i]] = np.hstack ((gobs[var_list[i]], o[var_list[i]]))

            gpredict[var_list[i]], gobs[var_list[i]] = gpredict[var_list[i]][:, 1 :], gobs[var_list[i]][:, 1 :]

            gprediction1[var_list[i]] = np.hstack ((gprediction1[var_list[i]], gpredict[var_list[i]]))
            gobserved1[var_list[i]] = np.hstack ((gobserved1[var_list[i]], gobs[var_list[i]]))

    for i in range(len(var_list)):
        gprediction1[var_list[i]] = gprediction1[var_list[i]][:, 1 :]
        gobserved1[var_list[i]] = gobserved1[var_list[i]][:, 1 :]
        gmse_test[var_list[i]], gmse_train[var_list[i]] = gmse_test[var_list[i]][1 :, :], gmse_train[var_list[i]][1 :, :]

    df,df1,df2,df3={},{},{},{}
    for i in range(len(var_list)):
        df[var_list[i]] = pd.DataFrame (gmse_train[var_list[i]])
        df1[var_list[i]] = pd.DataFrame (gmse_test[var_list[i]])
        df2[var_list[i]] = pd.DataFrame (gprediction1[var_list[i]])
        df3[var_list[i]] = pd.DataFrame (gobserved1[var_list[i]])


        df[var_list[i]].to_csv(fold+var_list[i]+'_tau_'+str(tau)+'_'+week+'_'+'_'+mgate+'_cgate_'+cgate+'_fluxgate_'+fluxgate+' fr_msetrain_MSEloss_hiddensize_tau.csv') #header=headerd
        df1[var_list[i]].to_csv(fold+var_list[i]+'_tau_'+str(tau)+'_'+week+'_'+'_'+mgate+'_cgate_'+cgate+'_fluxgate_'+fluxgate+' fr_msetest_MSEloss_hiddensize_tau.csv')
        df2[var_list[i]].to_csv(fold+var_list[i]+'_tau_'+str(tau)+'_'+week+'_'+'_'+mgate+'_cgate_'+cgate+'_fluxgate_'+fluxgate+' fr_prediction_MSEloss_hiddensize_tau.csv')
        df3[var_list[i]].to_csv(fold+var_list[i]+'_tau_'+str(tau)+'_'+week+'_'+'_'+mgate+'_cgate_'+cgate+'_fluxgate_'+fluxgate+' fr_observed_MSEloss_hiddensize_tau.csv')

    print('Saved all the files on LEO')
    print('Your LSTM model has been trained..')





if __name__=='__main__':

    seq_length = 30  #
    num_classes = 1  # number of outputs
    lr =.01 #learning rate .001
    num_epochs =40 #number of epochs

    resolutions = [0.012] #1 week
    fr = [np.arange(0.25,0.75,res) for res in resolutions]
    week=[f'{i}week' for i in range(1,2)]


    folder ='/home/...../LSTM/Results/Results_1week/'

    method=['regLSTM','mLSTM(tanh)']
    mt=1 #method index
    grad=['yes','no']
    gr=1 #add gradient of solutes in the LSTM acheitecture
    fluxgateid = 1 #fluxgate id 1==NO, 0==YES
    cgateid = 1 #all input SOLUTE gradients gate id 1==NO, 0==YES

    H=[32,45,50,55,60,64,68,70,73,75,78,80,83,85,87,90,93,95,98,100,105,110,115,120,125,128] #hidden layer size

    header = [f'h={hval}' for hval in H]

    varsup=[41,36,35,29,37,47,20,40,27,21]
    var_listsup=['Ca','Mg','Na','Cl','Al','Fe','DOC','K','SO4','NO3']
    inputvarsup=[[68,41],[68,41,36],[68,41,36,29,35],[68,36,35,29],[68,47,20,37],[68,37,20,29,47],[68,47,20],[68,47,20,40],[68,21,41,27],[68,27,20,29,21]]  #flow #test
    invar_listsup=[['Q','Ca'],['Q','Ca','Mg'],['Q','Ca','Mg','Cl','Na'],['Q','Mg','Na','Cl'],
                   ['Q','Fe','DOC','Al'],['Q','Al','DOC','Cl','Fe'],['Q','Fe','DOC'],['Q','Fe','DOC','K'],
                   ['Q','NO3','Ca','SO4'],['Q','SO4','DOC','Cl','NO3']]



    print('Output = ', var_listsup)
    print('week '+str(week))
    print(folder)
    print('lr='+str(lr))

    u,v={},{}
    for i in range (len (varsup)) :
        print (invar_listsup[i])
        r=utils_fg.dataretrive(varsup[i], inputvarsup[i],num_classes)
        u[var_listsup[i]], v[var_listsup[i]] = r, r[:,0]

    process1 = allvar(u, v, var_listsup, num_classes, H, num_epochs, fr[0], inputvarsup, invar_listsup, lr, header, seq_length, num_classes,device, grad[gr], method[mt],grad[cgateid],grad[fluxgateid], week[0], folder)
