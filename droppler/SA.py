#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  RRN_attention.py
#  
#  Copyright 2019 Gabriel Orlando <orlando.gabriele89@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
import torch
from torch import nn
import torch as t
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class attention(nn.Module):

	def __init__(self, nfea, heads, num_layers,outsize):
		super(attention, self).__init__()

		self.rnn=nn.GRU(input_size=nfea, hidden_size= heads, bidirectional=True, batch_first=True, num_layers= num_layers, dropout=0.1)
		#self.rnn=nn.LSTM(input_size=nfea, hidden_size= heads, bidirectional=True, batch_first=True, num_layers= num_layers, dropout=0.1)
		self.softmax= t.nn.Softmax(dim=1)
		ACTIVATION = t.nn.Tanh()
		
		self.final_linear= nn.Sequential(t.nn.Dropout(0.2),
										ACTIVATION, 
										nn.Linear(heads*2, outsize, bias=True))

	def forward(self, x, xlens):
		x,h =self.rnn(x)
		x=pad_packed_sequence(x,batch_first=True)[0]
		mask=masking(x,xlens)
		out=x.masked_fill(mask,0)

		out1=out.contiguous().view((out.shape[0]*out.shape[1],out.shape[2]))
		#out1 = t.cat([out1, p], 1)
		out1 = self.final_linear(out1)
		#out1=self.final_linear(out1.transpose(0,1).unsqueeze(0)).squeeze(0).transpose(0,1)
		out1=out1.view(out.shape[0],out.shape[1],-1)

		mask_soft = (torch.arange(out1.shape[1]).to(out1.device)[None, :] < xlens[:, None]).unsqueeze(2).repeat(1,1,out1.shape[2])

		out1=out1.masked_fill(~mask_soft,-float("inf"))
		out1=self.softmax(out1)
		return out1
		
class prediction(nn.Module):

	def __init__(self, nfea, hidden_size, num_layers,outsize):
		super(prediction, self).__init__()
		ACTIVATION = t.nn.Tanh()
		#self.rnn=nn.LSTM(input_size=nfea, hidden_size= hidden_size, bidirectional=True, batch_first=True, num_layers= num_layers, dropout=0.1)

		self.rnn = nn.GRU(input_size=nfea, hidden_size=hidden_size , bidirectional=True, batch_first=True, num_layers= num_layers, dropout=0.1)
		#self.finalPred = t.nn.Sequential(t.nn.Dropout(0.2), ACTIVATION, InceptionSimple(hidden_size*2, shrink=1), ACTIVATION,)
		
		#self.finalPred = t.nn.Sequential(ResidualBlock(t.nn.Conv1d(hidden_size*2, hidden_size*2, 3, padding=1)),  ACTIVATION, t.nn. Conv1d(hidden_size*2, 1, 1, padding=0), ACTIVATION)
		#self.finalPred = nn.Sequential( t.nn.InstanceNorm1d(hidden_size*2), ACTIVATION, t.nn.Conv1d(hidden_size*2, hidden_size, 3, padding=1),  t.nn.InstanceNorm1d(hidden_size), ACTIVATION, t.nn.Conv1d(hidden_size, 1, 3, padding=1), ACTIVATION)
		self.finalPred = nn.Sequential(	t.nn.Dropout(0.2),
										ACTIVATION, 
										nn.Linear(hidden_size*2, outsize, bias=True),
										ACTIVATION)
							
	def forward(self,x,xlens):
		x, h =self.rnn(x)
		x=pad_packed_sequence(x,batch_first=True)[0]
		mask=masking(x,xlens)
		out=x.masked_fill(mask,0)
		out2=out.contiguous().view((out.shape[0]*out.shape[1],out.shape[2]))
		out1 = self.finalPred(out2)

		out1=out1.view(out.shape[0],out.shape[1],-1)

		return out1#, out2
		
class selfatt_RRN(nn.Module):
	
	def __init__(self, nfea, nCond, hidden_size, num_layers, heads, name, outsize=10, condition_mix_neurons=10):
		super(selfatt_RRN, self).__init__()
		self.embedding = t.nn.Embedding(21, nfea, padding_idx = 0)
		self.pred = prediction(nfea, hidden_size=hidden_size, num_layers=num_layers, outsize=outsize)
		self.att  = attention(nfea, heads=heads, num_layers = num_layers, outsize=outsize)
		
		self.condNNpre  = t.nn.Sequential(t.nn.Linear(nCond, condition_mix_neurons),
										  t.nn.Dropout(0.2),
										  t.nn.Tanh())
										  
		self.condNNpost = t.nn.Sequential(t.nn.Linear(condition_mix_neurons+outsize, 10),
										  t.nn.Tanh(),
										  t.nn.Linear(10, 1))
	
		self.name=name
		self.apply(init_weights)

	def forward(self, x, xc, xlens):
		x = self.embedding(x)
		#if type(xlens)==type(None):
		#	xlens=torch.Tensor([x.shape[1]]*x.shape[0]).type(torch.LongTensor)
		x = pack_padded_sequence(x, xlens.cpu(), batch_first=True, enforce_sorted=False)
		p = self.pred(x, xlens)
		a = self.att(x, xlens)

		out=[]
		for i in range(p.shape[2]):
			out+=[torch.bmm(torch.unsqueeze(p[:,:,i],1),torch.unsqueeze(a[:,:,i],-1)).squeeze(2)]

		out = torch.cat(out,dim=1)
		outC = self.condNNpre(xc)
		out = t.cat([outC, out],dim=1)
		out=self.condNNpost(out)
		return out.squeeze(), a

		
def masking(X,X_len):
	maxlen = X.size(1)
	mask = torch.arange(maxlen).to(X.device)[None, :] < X_len[:, None]
	mask=torch.unsqueeze(mask,2)
	mask=mask.expand(-1,-1,X.shape[2])
	return ~mask

def pad_sequence(sequences, batch_first=True):
	max_size = sequences[0].size()
	max_len, trailing_dims = max_size[0], max_size[1:]
	prev_l = max_len
	if batch_first:
		out_dims = (len(sequences), max_len) + trailing_dims
	else:
		out_dims = (max_len, len(sequences)) + trailing_dims
	out_variable = sequences[0].new(*out_dims).zero_()
	lengths=[]
	for i, variable in enumerate(sequences): 
		length = len(sequences[i])         
		lengths.append(length)
		# temporary sort check, can be removed when we handle sorting internally
		if prev_l < length:
			raise ValueError("lengths array has to be sorted in decreasing order")
		prev_l = length
		# use index notation to prevent duplicate references to the variable
		if batch_first:
			out_variable[i, :length, ...] = variable
		else:
			out_variable[:length, i, ...] = variable
	return out_variable, lengths

def sortForPaddingTemp(X, Xss, Y):
	assert len(X) == len(Y) == len(Xss)
	tmp = []
	i = 0
	while i < len(X):
		tmp.append((X[i], Xss[i], Y[i], i))
		i+=1
	tmp = sorted(tmp, key=lambda x: len(x[0]), reverse=True)
	i = 0
	Xt = []
	Xsst = []
	Yt = []
	order = []
	while i < len(tmp):
		Xt.append(t.tensor(tmp[i][0], dtype=t.long))
		Xsst.append(t.tensor(tmp[i][1], dtype=t.float))
		Yt.append(tmp[i][2])
		order.append(tmp[i][-1])
		i+=1
	return Xt, Xsst, t.tensor(Yt, dtype=t.float), order #lists of tensors

def sortForPadding(X, Y):
	assert len(X) == len(Y)
	tmp = []
	i = 0
	while i < len(X):
		tmp.append((X[i], Y[i], i))
		i+=1
	tmp = sorted(tmp, key=lambda x: len(x[0]), reverse=True)
	i = 0
	Xt = []
	Yt = []
	order = []
	while i < len(tmp):
		Xt.append(t.tensor(tmp[i][0], dtype=t.long))
		Yt.append(tmp[i][1])
		order.append(tmp[i][-1])
		i+=1
	return Xt, t.tensor(Yt, dtype=t.float), order #lists of tensors

def unpad(x, l):
	preds=[]
	for i in xrange(len(x)):
		preds.append(x[i][:l[i]])
	return preds
	
def reconstruct(x, l):
	preds=[]
	pos = 0
	for p in l:
		tmp = x[pos:pos+p]
		preds.append(tmp)
		pos = p
	return preds	


def init_weights(m):
	if isinstance(m, t.nn.Conv2d) or isinstance(m, t.nn.Conv1d) or isinstance(m, t.nn.Linear) or isinstance(m, t.nn.Bilinear):
		#print("Initializing weights...", m.__class__.__name__)
		#t.nn.init.normal_(m.weight, 0, 0.01)
		t.nn.init.xavier_uniform_(m.weight)
		if type(m.bias)!=type(None):
			m.bias.data.fill_(0.8)
	elif isinstance(m, t.nn.Embedding):
		#print("Initializing weights...", m.__class__.__name__)
		#t.nn.init.normal_(m.weight, 0, 0.1)
		t.nn.init.xavier_uniform_(m.weight)
	elif type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
		#print("Initializing weights...", m.__class__.__name__)
		for name, param in m.named_parameters():
			if 'weight_ih' in name:
				#t.nn.init.normal_(param.data, 0, 0.01)
				torch.nn.init.xavier_uniform_(param.data)
			elif 'weight_hh' in name:
				#torch.nn.init.normal(param.data)
				torch.nn.init.orthogonal_(param.data)
			elif 'bias' in name:
				param.data.fill_(0.8)


def main(args):
    return 0

if __name__ == '__main__':
	#a=selfatt_tua(nfea=61)
	import cProfile
	c=torch.Tensor(np.random.randn(15,500,61))
	a=selfatt_RRN(61,dev=t.device('cpu'),compress_initial_features=False)
	#c=c.permute(1,0)
	parameters = list(a.parameters())
	p = []
	for i in parameters:
		p+= list(i.data.cpu().numpy().flat)
	#print len(p)
	lens=torch.Tensor(np.random.randint(1,500,(15))).type(torch.LongTensor)
	#cProfile.run('a(c.to("cpu"),lens)')
	#asd
	b= a(c.to("cpu"),lens)
	print(b)


