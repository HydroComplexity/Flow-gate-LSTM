
########################################Multiprocessing#########################################

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.optim as optim
import torch
from torch import nn
from sklearn.metrics import mean_squared_error as MSE
from torch.multiprocessing import Process
import os
import sys
from tqdm import tqdm
import utils_fglstm as utils_fg

device = sys.argv[1]
device  = 'cpu'
print('running on CPU')

seed=42
torch.manual_seed(seed)

def allvar(training_set,var_list,tau,H,num_epochs,fr,inputvar,invar_list,learning_rate,header,seq_length,num_classes,device,mgate,fold):

    process_id=os.getpid()
    print('######################################')
    print('Process ID'+str(process_id))
    print('######################################')
    print(mgate)

    ngpredt_train_var = []      #normalised values predicted training set: for all hidden_size (number of tau in column)
    ngpredt_var = []             #normalised values predicted testing set: for all hidden_size (number of tau in column)
    ngobs_var = []              #normalised values observed testing set: for all hidden_size (number of tau in column)
    ngobs_train_var = []        #normalised values observed training set: for all hidden_size (number of tau in column)

    mse_mlstmts = np.empty (len (H))    #np.empty ((len (tau), len (H)))
    mse_mlstmtr = np.empty (len (H))
    hk = 0              #initialization of hidden size vector

    training_seti=training_set[1:,:]
    slp = utils_fg.getgrad (training_set[:,1])
    training_seti = np.hstack ((training_seti, slp.reshape ((len (slp), 1))))
    training_set=training_seti

    for hidden_size in tqdm(H):
        nk=tau

        input_size1 = len(inputvar)*(nk)+1        #number of inputs
        diffvar1=len(inputvar)*(nk)+1

        input_size = len(inputvar)*(nk)        #number of inputs


        sc = MinMaxScaler()
        training_data = sc.fit_transform(training_set)  #normalisation

        x, y = utils_fg.sliding_windows(training_data, seq_length)

        train_size = int(len(y) * fr)
        test_size = len(y) - train_size


        trainX = torch.Tensor(np.array(x[:train_size,:,:])).to(device)
        trainY = torch.Tensor(np.array(y[:train_size,:])).to(device)

        testX = torch.Tensor(np.array(x[train_size:,:,:])).to(device)
        testY = torch.Tensor(np.array(y[train_size:,:])).to(device)


        class LSTM_cell_flow_gate (torch.nn.Module) :
            """
            A simple LSTM cell network for educational AI-summer purposes
            """
            def __init__(self, input_length=input_size, hidden_length=hidden_size,output_length=num_classes,device=device) :
                super (LSTM_cell_flow_gate, self).__init__ ()
                self.input_length = input_length
                self.hidden_length = hidden_length
                self.output_length = output_length

                # forget gate components
                self.linear_forget_w1 = nn.Linear (self.input_length, self.hidden_length, bias=False).to(device)
                self.linear_forget_r1 = nn.Linear (self.hidden_length, self.hidden_length, bias=False).to(device)
                self.linear_forget_l1 = nn.Linear (self.hidden_length, self.hidden_length, bias=False).to(device)

                self.sigmoid_forget = nn.Sigmoid ().to(device)
                    # input gate components
                self.linear_gate_w2 = nn.Linear (self.input_length, self.hidden_length, bias=False).to(device)
                self.linear_gate_r2 = nn.Linear (self.hidden_length, self.hidden_length, bias=False).to(device)
                self.linear_gate_l2 = nn.Linear (self.hidden_length, self.hidden_length, bias=False).to(device)
                self.linear_gate_d2 = nn.Linear (1, self.hidden_length, bias=False).to (device)    #dilution gate 1 because using only gradient
                # self.linear_gate_d3 = nn.Linear (1, self.hidden_length, bias=False).to (device)  # dilution gate 1 because using only gradient
                self.linear_gate_flux2 = nn.Linear (1, self.hidden_length, bias=False).to (device)  # flux gate
                self.linear_gate_c2 = nn.Linear (1, self.hidden_length, bias=False).to (device)  # solute gate
                self.sigmoid_gate = nn.Sigmoid ().to(device)
                    # cell memory components
                self.linear_gate_w3 = nn.Linear (self.input_length, self.hidden_length, bias=False).to(device)
                self.linear_gate_r3 = nn.Linear (self.hidden_length, self.hidden_length, bias=False).to(device)
                self.linear_gate_l3 = nn.Linear (self.hidden_length, self.hidden_length, bias=False).to(device)
                self.activation_gate = nn.Tanh ().to(device)
                    # out gate components
                self.linear_gate_w4 = nn.Linear (self.input_length, self.hidden_length, bias=False).to(device)
                self.linear_gate_r4 = nn.Linear (self.hidden_length, self.hidden_length, bias=False).to(device)
                self.linear_gate_l4 = nn.Linear (self.hidden_length, self.hidden_length, bias=False).to(device)
                self.sigmoid_hidden_out = nn.Sigmoid ().to(device)
                self.activation_final = nn.Tanh ().to(device)

            def forget(self, x, h, c) :
                idx=np.arange (0, len (invar_list), 1)
                y=torch.take(x, torch.tensor(np.array(idx)))
                y=torch.reshape (y, (1, len(invar_list)))
                x1 = self.linear_forget_w1 (y)
                h1 = self.linear_forget_r1 (h)
                c1 = self.linear_forget_l1 (c)
                return self.sigmoid_forget (x1 + h1 + c1)

            def input_gate(self, x, h, c) :
                idx=np.arange (0, len (invar_list), 1)
                y=torch.take(x, torch.tensor(np.array(idx)))
                y=torch.reshape (y, (1, len(invar_list)))
                x_temp = self.linear_gate_w2 (y)
                h_temp = self.linear_gate_r2 (h)
                c_temp = self.linear_gate_l2 (c)
                if mgate == 'regLSTM':
                    return self.sigmoid_gate (x_temp + h_temp + c_temp)
                if mgate == 'mLSTM(tanh)':
                    ior = self.sigmoid_gate (x_temp + h_temp + c_temp)  # changed original sig
                    d = self.linear_gate_d2 ((torch.reshape (x[0][len (inputvar)], (1, 1))))
                    icr = ior + self.activation_gate (d)  # + self.activation_gate(d1)# original tanh
                    return icr


            def cell_memory_gate(self, i, f, x, h, c_prev) :
                idx=np.arange (0, len (invar_list), 1)
                y=torch.take(x, torch.tensor(np.array(idx)))
                y=torch.reshape (y, (1, len(invar_list)))
                x1 = self.linear_gate_w3 (y)
                h1 = self.linear_gate_r3 (h)
                k = self.activation_gate (x1 + h1)
                g = k * i
                c=f*c_prev
                c_next = g + c
                return c_next
            def out_gate(self, x, h, c) :
                idx=np.arange (0, len (invar_list), 1)
                y=torch.take(x, torch.tensor(np.array(idx)))
                y=torch.reshape (y, (1, len(invar_list)))
                x1 = self.linear_gate_w4 (y)
                h1 = self.linear_gate_r4 (h)
                c1 = self.linear_gate_l4 (c)
                return self.sigmoid_hidden_out (x1 + h1 + c1)


            def forward(self, x, tuple_in) :   #x is input_t  tuple_in=h_t,c_t
                (h, c_prev) = tuple_in
                # Equation 1. input gate
                i = self.input_gate (x, h, c_prev)
                # Equation 2. forget gate
                f = self.forget (x, h, c_prev)
                # Equation 3. updating the cell memory
                c_next = self.cell_memory_gate (i, f, x, h, c_prev)
                # Equation 4. calculate the main output gate
                o = self.out_gate (x, h, c_next)
                # Equation 5. produce next hidden output
                h_next = o * self.activation_final (c_next)

                return h_next, c_next



        class Sequence (nn.Module) :
            def __init__(self, LSTM=True, custom=True, device = device) :
                super (Sequence, self).__init__ ()
                self.LSTM = LSTM
                self.device = device
                if LSTM :
                    if custom :
                        print ("LSTM cell implementation...")
                        self.rnn1 = LSTM_cell_flow_gate(input_size, hidden_size, num_classes,device).to(device)      #inputlenth and hidden size

                    else :
                        print ("Official PyTorch LSTM cell implementation...")
                        self.rnn1 = nn.LSTMCell (seq_length, hidden_size).to(device)

                self.linear = nn.Linear(hidden_size, num_classes).to(device)


            def forward(self, input, future=0) :
                outputs = torch.empty(1,num_classes, device=self.device)    #number of class is input var
                h_t = torch.zeros (1, hidden_size, dtype=torch.float, device=self.device)       #data x hiddenzsize (50) should be H X 1
                c_t = torch.zeros (1, hidden_size, dtype=torch.float,device=self.device)       #data x hiddenzsize (50) should be H X 1

                for i, input_t in enumerate(input.chunk(input.size(0), dim=0)) :    #should be var x seq_length
                    input_t=torch.squeeze(input_t,0)
                    h_seq=[]
                    for j, input_t1 in enumerate(input_t.chunk(input_t.size(0), dim=0)) :

                        input_t1= torch.squeeze(input_t1, 0)
                        input_t1=torch.reshape(input_t1,(1,len(inputvar)*nk+1))
                        if self.LSTM :
                            h_t, c_t = self.rnn1 (input_t1, (h_t, c_t))

                        else :
                            h_t = self.rnn1 (input_t1, h_t)

                        h_seq.append(h_t.unsqueeze(0))

                    h_seq=torch.cat(h_seq,dim=0)
                    h_seq=h_seq.transpose(0,1).contiguous()
                    output = self.linear (h_seq[:,-1,:])  #h_t2
                    outputs=torch.cat((outputs,output),0)

                    for i in range (future) :
                        if self.LSTM :
                            h_t, c_t = self.rnn1 (input_t1, (h_t, c_t))
                        else :
                            h_t = self.rnn1 (input_t1, h_t)

                return outputs[1:]



        if __name__ == '__main__' :
            input = trainX[:,:,input_size1+num_classes-diffvar1:]                                     #trainX row from 3 and column till last -1

            if mgate == 'regLSTM' :
                print ('Not using flow gate regLSTM')
            else :
                print ('Using flow gate mLSTM')


            target = trainY[:,0:num_classes]                                       #trainY row from 3rd and column from 1 to end
            test_input = testX[:,:,input_size1+num_classes-diffvar1:]                                    #till row 3 and column till last -1
            test_target = testY[:,0:num_classes]                                  # till row 3 and column from 1 to last

            seq = Sequence (LSTM=True, custom=True).to(device)
            seq.float ()
            criterion = nn.MSELoss().to(device)

            optimizer = optim.Adam(seq.parameters (), lr=learning_rate)
                # begin to train
            lss = np.ones ((num_epochs, 1))
            for i in range (num_epochs) :
                seq.train()
                optimizer.zero_grad ()
                out = seq(input)                           #trainX      data x sequencelenth x number of variable
                loss = criterion (out, target)                  #training set target=trainY
                if i % 4 == True :
                    print ('epoch:', i ,'loss:', loss.item ())
                if loss.item()<=.0005:
                    break
                lss[i]=loss.item()
                if i>=2:
                    lss1=lss[i]-lss[i-2]
                    if abs(lss1)<=0.0005:
                        break
                loss.backward ()
                optimizer.step ()

            with torch.no_grad () :  # no gradient calculations
                seq.eval ()
                future = 0
                pred = seq (test_input, future=future)
                pred_train = seq (input, future=future)
                loss1 = criterion (pred_train, target)
                loss = criterion (pred, test_target)
                print ('train loss:', loss1.item ())
                print ('test loss:', loss.item ())


            yy = pred.detach().cpu().numpy()
            tr = pred_train.detach().cpu().numpy ()
            target=target.cpu()
            test_target=test_target.cpu()
                # draw the result
                #lencorr1 is for the inverse transformation
            train_trfmp=utils_fg.lencorr1(tr,len(y))
            train_trfmo=utils_fg.lencorr1(target,len(y))
            test_trfmp=utils_fg.lencorr1(yy,len(y))
            test_trfmo=utils_fg.lencorr1(test_target,len(y))

            ntrain_predict=train_trfmp[:len(tr)]              #normalised values
            ntrainY_plot=train_trfmo[:len(target)]
            ntest_predict=test_trfmp[:len(yy)]
            ntestY_plot=test_trfmo[:len(test_target)]

            if np.isnan(ntrain_predict).any():
                mse_mlstmtr[hk]=11111
                mse_mlstmts[hk]=11111
                print ('nan in prediction')
            else:
                mse_mlstmtr[hk] = MSE (ntrainY_plot, ntrain_predict, multioutput='raw_values')
                mse_mlstmts[hk]=MSE(ntestY_plot,ntest_predict,multioutput='raw_values')


            #prediction is happening at different time
            #lencorr2 is correcting the time step displacement because of the tau
            ntrain_ip=utils_fg.lencorr2(ntrain_predict,train_size,seq_length)
            ntrain_io=utils_fg.lencorr2(ntrainY_plot,train_size,seq_length)
            ntest_ip=utils_fg.lencorr2(ntest_predict,test_size,0)
            ntest_io=utils_fg.lencorr2(ntestY_plot,test_size,0)

            ngpredt_var.append (ntest_ip)                      #for every tau
            ngpredt_train_var.append (ntrain_ip)               #for every tau
            ngobs_var.append (ntest_io)                         #for every tau
            ngobs_train_var.append (ntrain_io)                 #for every tau

        if hk+1<len(H):
            print('######################################################################################')
            print("####################################### Hidden Size ",str(H[hk]) , "#####################################")
            print("####################", "  ", "Now starting Hidden Size =",str(H[hk+1]), "#######################")
            print("######################################################################################")
        hk=hk+1


    xp, yp=np.min (np.min(mse_mlstmtr)), np.where (mse_mlstmtr == np.min (np.min(mse_mlstmtr)))                  #training va
    ntr_pr=ngpredt_train_var[yp[0][0]]
    nts_pr=ngpredt_var[yp[0][0]]
    ntr_obs=ngobs_train_var[yp[0][0]]
    nts_obs=ngobs_var[yp[0][0]]

    #Save files
    nprediction1 = np.concatenate ((ntr_pr, nts_pr))
    nobserved1 = np.concatenate ((ntr_obs, nts_obs))

    df2=pd.DataFrame(nprediction1)
    df3=pd.DataFrame(nobserved1)
    df4=pd.DataFrame(np.transpose([mse_mlstmtr],(0,1)))
    df5=pd.DataFrame(np.transpose([mse_mlstmts],(0,1)))

    df2.to_csv(fold+var_list+'_'+str(invar_list)+'_tau_'+str(tau)+'_fr_'+str(fr)+'_'+mgate+'_nprediction_MSEloss_hiddensize_tau.csv')
    df3.to_csv(fold+var_list+'_'+str(invar_list)+'_tau_'+str(tau)+'_fr_'+str(fr)+'_'+mgate+'_nobserved_MSEloss_hiddensize_tau.csv')
    df4.to_csv(fold+var_list+'_'+str(invar_list)+'_tau_'+str(tau)+'_fr_'+str(fr)+'_'+mgate+'_mse_traindata_MSEloss_hiddensize_tau.csv',header=header)
    df5.to_csv (fold+var_list+'_'+str(invar_list)+'_tau_'+str(tau)+'_fr_'+str(fr)+'_'+mgate+'_mse_testdata_MSEloss_hiddensize_tau.csv',header=header)






############################### model Input ###################################
seq_length = 30
num_classes = 1  # number of outputs

fr=[0.5]  #[0.25,0.37,0.5,0.67,0.75] 
num_epochs =40


folder='/home/.../out_file/'

method=['regLSTM','mLSTM(tanh)']
ind=0


H = [32,40,45,50,55,60,64,68,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,150,160]


header = [f'h={hval}' for hval in H]


lr =.01 #learning rate .001
print('lr='+str(lr))



var=[41,36,35,29,37,47,20,27,21]
var_list=['Ca','Mg','Na','Cl','Al','Fe','DOC','SO4','NO3']
#
######### based on the U1+U2+S component from PID (Partial information decomposition) ###########
####### adding solutes iteratively ########
######## Can be done with the parallel processing and searching for the dependent/input solute vector for LSTM with iteratively checking r^2 values########
inputvar2=[[68,41],[68,36],[68,35],[68,29],[68,37],[68,47],[68,20],[68,27],[68,21]]
invar_list2=[['Q','Ca'],['Q','Mg'],['Q','Na'],['Q','Cl'],['Q','Al'],['Q','Fe'],['Q','DOC'],['Q','SO4'],['Q','NO3']]
inputvar3=[[68,36,41],[68,41,36],[68,29,35],[68,35,29],[68,47,37],[68,37,47],[68,47,20],[68,21,27],[68,27,21]]
invar_list3=[['Q','Mg','Ca'],['Q','Ca','Mg'],['Q','Cl','Na'],['Q','Na','Cl'],['Q','Fe','Al'],['Q','Al','Fe'],
             ['Q','Fe','DOC'],['Q','NO3','SO4'],['Q','SO4','NO3']]
inputvar4=[[68,36,37,41],[68,41,29,36],[68,29,41,35],[68,35,36,29],[68,47,20,37],[68,37,20,47],[68,47,37,20],[68,21,41,27],[68,27,20,21]]
invar_list4=[['Q','Mg','Al','Ca'],['Q','Ca','Cl','Mg'],['Q','Cl','Ca','Na'],['Q','Na','Mg','Cl'],['Q','Fe','DOC','Al'],
             ['Q','Al','DOC','Fe'],['Q','Fe','Al','DOC'],['Q','NO3','Ca','SO4'],['Q','SO4','DOC','NO3']]
inputvar5=[[68,36,37,35,41],[68,41,29,35,36],[68,29,41,36,35],[68,35,36,41,29],[68,47,20,41,37],[68,37,20,29,47],[68,47,37,29,20],[68,21,41,29,27],[68,27,20,29,21]]
invar_list5=[['Q','Mg','Al','Na','Ca'],['Q','Ca','Cl','Na','Mg'],['Q','Cl','Ca','Mg','Na'],['Q','Na','Mg','Ca','Cl'],
             ['Q','Fe','DOC','Ca','Al'],['Q','Al','DOC','Cl','Fe'],['Q','Fe','Al','Cl','DOC'],['Q','NO3','Ca','Cl','SO4'],['Q','SO4','DOC','Cl','NO3']]
inputvar6=[[68,36,37,35,29,41],[68,41,29,35,37,36],[68,29,41,36,37,35],[68,35,36,41,21,29],[68,47,20,41,36,37],[68,37,20,29,35,47],[68,47,37,29,35,20],[68,21,41,29,36,27],[68,27,20,29,35,21]]
invar_list6=[['Q','Mg','Al','Na','Cl','Ca'],['Q','Ca','Cl','Na','Al','Mg'],['Q','Cl','Ca','Mg','Al','Na'],['Q','Na','Mg','Ca','NO3','Cl'],
              ['Q','Fe','DOC','Ca','Mg','Al'],['Q','Al','DOC','Cl','Na','Fe'],['Q','Fe','Al','Cl','Na','DOC'],['Q','NO3','Ca','Cl','Mg','SO4'],['Q','SO4','DOC','Cl','Na','NO3']]
inputvar7=[[68,36,37,35,29,47,41],[68,41,29,35,37,20,36],[68,29,41,36,37,20,35],[68,35,36,41,21,20,29],[68,47,20,41,36,35,37],[68,37,20,29,35,36,47],[68,47,37,29,35,36,20],[68,21,41,29,36,35,27],[68,27,20,29,35,47,21]]
invar_list7=[['Q','Mg','Al','Na','Cl','Fe','Ca'],['Q','Ca','Cl','Na','Al','DOC','Mg'],['Q','Cl','Ca','Mg','Al','DOC','Na'],['Q','Na','Mg','Ca','NO3','DOC','Cl'],
              ['Q','Fe','DOC','Ca','Mg','Na','Al'],['Q','Al','DOC','Cl','Na','Mg','Fe'],['Q','Fe','Al','Cl','Na','Mg','DOC'],['Q','NO3','Ca','Cl','Mg','Na','SO4'],['Q','SO4','DOC','Cl','Na','Fe','NO3']]


inputvar1=[inputvar2,inputvar3,inputvar4,inputvar5,inputvar6,inputvar7]
invar_list1=[invar_list2,invar_list3,invar_list4,invar_list5,invar_list6,invar_list7]

print('Output='+str(var_list))
print('learning rate='+str(lr))
print('seq length='+str(seq_length))
print('lenth of fr '+str(len(fr)))

print('method='+method[ind])
print(folder)


for i in range (len (var)) :
    for j in range(len(inputvar1)):
        print (invar_list1[j])
        inputvar=inputvar1[j]
        invar_list=invar_list1[j]
        u=utils_fg.dataretrive(var[i], inputvar[i], num_classes)


        for k in range(len(fr)):
            process1 = Process (target=allvar, args=(u, var_list[i], num_classes, H, num_epochs, fr[k], inputvar[i], invar_list[i], lr, header, seq_length, num_classes, device,method[ind],folder))
            process1.start ()


print ("multiprocessing completed only hidden state")
