import torch
from torch import nn
import numpy as np
import math


class LSTM_cell_flow_gate (torch.nn.Module) :
    """
    A simple LSTM cell network for educational AI-summer purposes
    """

    def __init__(self, input_length=256, hidden_length=256, output_length=1, stshy=0, device='cuda', mgate='regLSTM',cgate='no',fluxgate='no',grad='yes',inputvar=[]) :
        super (LSTM_cell_flow_gate, self).__init__ ()
        self.input_length = input_length
        self.hidden_length = hidden_length
        self.output_length = output_length
        self.inputvar  = inputvar
        self.mgate = mgate
        self.cgate = cgate
        self.fluxgate = fluxgate
        self.grad=grad
        self.device=device
        self.stshy=stshy

        self.linear_gate_w1 = nn.Linear (self.input_length, self.hidden_length, bias=False).to (device)
        self.linear_gate_r1 = nn.Linear (self.hidden_length, self.hidden_length, bias=False).to (device)
        self.linear_gate_l1 = nn.Linear (self.hidden_length, self.hidden_length, bias=False).to (device)
        self.sigmoid_gate = nn.Sigmoid ().to (device)

        self.linear_gate_w2 = nn.Linear (self.input_length, self.hidden_length, bias=False).to (device)
        self.linear_gate_r2 = nn.Linear (self.hidden_length, self.hidden_length, bias=False).to (device)
        self.linear_gate_l2 = nn.Linear (self.hidden_length, self.hidden_length, bias=False).to (device)
        self.linear_gate_d2 = nn.Linear (1, self.hidden_length, bias=False).to (device)  # dilution gate 1 because using only gradient
        # self.linear_gate_c2 = nn.Linear (1, self.hidden_length, bias=False).to (device)  # target chemical gradient gate
        # self.linear_gate_flux2 = nn.Linear (1, self.hidden_length, bias=False).to (device)  # target flux grad gate
        self.sigmoid_gate = nn.Sigmoid ().to (device)

        self.linear_gate_w3 = nn.Linear (self.input_length, self.hidden_length, bias=False).to (device)
        self.linear_gate_r3 = nn.Linear (self.hidden_length, self.hidden_length, bias=False).to (device)
        self.linear_gate_l3 = nn.Linear (self.hidden_length, self.hidden_length, bias=False).to (device)
        self.activation_gate = nn.Tanh ().to (device)

        self.linear_gate_w4 = nn.Linear (self.input_length, self.hidden_length, bias=False).to (device)
        self.linear_gate_r4 = nn.Linear (self.hidden_length, self.hidden_length, bias=False).to (device)
        self.linear_gate_l4 = nn.Linear (self.hidden_length, self.hidden_length, bias=False).to (device)
        self.sigmoid_hidden_out = nn.Sigmoid ().to (device)
        self.activation_final = nn.Tanh ().to (device)

    def forget(self, x, h, c) :
        if self.grad == 'no' :
            idx = np.arange (0, len (self.inputvar)+self.stshy, 1)
            y = torch.take (x, torch.tensor (np.array (idx)).to(self.device))
            y = torch.reshape (y, (1, len (self.inputvar)+self.stshy))
            x1 = self.linear_gate_w1 (y)
        else :
            x1 = self.linear_gate_w1 (x)

        h1 = self.linear_gate_r1 (h)
        c1 = self.linear_gate_l1 (c)
        return self.sigmoid_gate (x1 + h1 + c1)

    def input_gate(self, x, h, c) :
        h_temp = self.linear_gate_r2 (h)
        c_temp = self.linear_gate_l2 (c)
        x_temp = self.linear_gate_w2 (x)

        if self.mgate == 'regLSTM':
            return self.sigmoid_gate (x_temp + h_temp + c_temp)
        if self.mgate == 'mLSTM(tanh)':
            ior = self.sigmoid_gate (x_temp + h_temp + c_temp)  # changed original sig

            d = self.linear_gate_d2 ((torch.reshape (x[0][len(self.inputvar)], (1, 1))))
            icr = ior + self.activation_gate (d)  # + self.activation_gate(d1)# original tanh
            return icr

    def cell_memory_gate(self, i, f, x, h, c_prev) :
        if self.grad == 'no' :
            idx = np.arange (0, len (self.inputvar)+self.stshy, 1)
            y = torch.take (x, torch.tensor (np.array (idx)).to(self.device))
            y = torch.reshape (y, (1, len (self.inputvar)+self.stshy))
            x1 = self.linear_gate_w3 (y)
        else :
            x1 = self.linear_gate_w3 (x)

        h1 = self.linear_gate_r3 (h)
        k = self.activation_gate (x1 + h1)
        g = k * i

        c = f * c_prev
        c_next = g + c
        return c_next

    def out_gate(self, x, h, c) :
        if self.grad == 'no' :
            idx = np.arange (0, len (self.inputvar)+self.stshy, 1)
            y = torch.take (x, torch.tensor (np.array (idx)).to(self.device))
            y = torch.reshape (y, (1, len (self.inputvar)+self.stshy))
            x1 = self.linear_gate_w4 (y)
        else :
            x1 = self.linear_gate_w4 (x)
        h1 = self.linear_gate_r4 (h)
        c1 = self.linear_gate_l4 (c)
        return self.sigmoid_hidden_out (x1 + h1 + c1)

    def forward(self, x, tuple_in) :  # x is input_t (Q,tar,dependent vari, gradQ, grad C) tuple_in=h_t,c_t
        (h, c_prev) = tuple_in
        i = self.input_gate (x, h, c_prev)
        f = self.forget (x, h, c_prev)
        c_next = self.cell_memory_gate (i, f, x, h, c_prev)
        o = self.out_gate (x, h, c_next)
        h_next = o * self.activation_final (c_next)
        return h_next, c_next


class Sequence (nn.Module) :

    def __init__(self,input_size, hidden_size, seq_length, num_classes, mgate,inputvar,nk, stshy=0,LSTM=True, custom=True, device='cuda', ) :
        super (Sequence, self).__init__ ()
        self.LSTM = LSTM
        self.device = device
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.inputvar = inputvar
        self.nk = nk
        self.stshy=stshy
        self.input_size=input_size
        if LSTM :
            if custom :
                print ("modified LSTM cell implementation...")
                self.rnn1 = LSTM_cell_flow_gate (input_size, hidden_size, num_classes, stshy, device, mgate,inputvar).to (device)  # inputlenth and hidden size
            else :
                print ("Official PyTorch LSTM cell implementation...")
                self.rnn1 = nn.LSTMCell (seq_length, hidden_size).to (device)
        self.linear = nn.Linear (hidden_size, num_classes).to (device)

    def forward(self, input, future=0) :  
        outputs = torch.ones (1, self.num_classes, device=self.device)  # number of class is input var
        h_t = torch.zeros (1, self.hidden_size, dtype=torch.float,device=self.device)  # data x hiddenzsize (50) should be H X 1
        c_t = torch.zeros (1, self.hidden_size, dtype=torch.float,device=self.device)  # data x hiddenzsize (50) should be H X 1
        for i, input_t in enumerate (input.chunk (input.size (0), dim=0)) :  # should be var x seq_length
            input_t = torch.squeeze (input_t, 0)
            h_seq = []
            for j, input_t1 in enumerate (input_t.chunk (input_t.size (0), dim=0)) :
                input_t1 = torch.squeeze (input_t1, 0)
                input_t1 = torch.reshape (input_t1, (1, len (self.inputvar) * (self.nk) + 1+self.stshy))  #
                if self.LSTM :
                    h_t, c_t = self.rnn1 (input_t1, (h_t, c_t))
                else :
                    h_t = self.rnn1 (input_t1, h_t)

                h_seq.append (h_t.unsqueeze (0))
            h_seq = torch.cat (h_seq, dim=0)
            h_seq = h_seq.transpose (0, 1).contiguous ()
            output = self.linear (h_seq[:, -1, :])  # h_t2
            outputs = torch.cat ((outputs, output), 0)
            for i in range (future) :
                if self.LSTM :
                    h_t, c_t = self.rnn1 (input_t1, (h_t, c_t))
                else :
                    h_t = self.rnn1 (input_t1, h_t)
        return outputs[1 :]



class RealNVP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RealNVP, self).__init__()

        # Define the neural network for the scale and translation parameters
        self.netin = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size * 2)
        )
        self.netout = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size * 2)
        )

    def forward(self, x):   #x=s
        # Split the input into two parts
        u1, u2 = x.chunk(2, dim=-1)

        # Pass the input through the neural network
        s2, t2 = self.netin(u2).chunk(2, dim=-1)

        # Apply the scale and translation parameters
        s2 = torch.sigmoid(s2)
        v1 = u1 * math.exp(s2 + t2)

        s1, t1 = self.netout(v1).chunk (2, dim=-1)
        s1 = torch.sigmoid (s1)

        v2 = u2 * math.exp (s1 + t1)

        # Concatenate the two parts of the input
        x = torch.cat([v1, v2], dim=-1)

        return x

