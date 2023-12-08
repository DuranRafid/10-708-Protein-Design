#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  untitled.py
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
from sklearn.metrics import roc_auc_score
from torch import nn
import os, copy, random, time, sys
from sys import stdout
import numpy as np
import torch as t
t.autograd.set_detect_anomaly(True)
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence,pad_sequence
from torch.utils.data import Dataset, DataLoader
	
class predictDataset(Dataset):
    
	def __init__(self, X, Xcond, lens):
		self.X = X 
		self.Xcond = Xcond
		self.lens = lens
		assert len(self.X) > 0
		
	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.Xcond[idx], self.lens[idx]

class PSPDataset(Dataset):
    
	def __init__(self, X, Xcond, Y, lens):
		self.Y = Y
		self.Xcond = Xcond
		self.X = X 
		self.lens = lens
		assert len(self.X) > 0
		
	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.Xcond[idx], self.Y[idx], self.lens[idx]

class NNwrapper():

	def __init__(self, model):
		self.model = model
	
	def fit(self, originalX, Xconditions, originalY, lens, epochs, batch_size, weight_decay, learning_rate, dev, save_model_every=10, LOG=False):
		os.system("mkdir -p models")
		self.model.train()
		if LOG:
			os.system("rm -rf ./logs")
			os.system("killall tensorboard")
			os.system("tensorboard --logdir='./logs' --port=6006 &")
			from pytorchUtils.logger import Logger, to_np, to_var
			logger = Logger('./logs')			
			os.system("firefox http://127.0.0.1:6006  &")
		########DATASET###########
		dataset = PSPDataset(originalX, Xconditions, originalY, lens)
		
		#######MODEL##############		
		parameters =list(self.model.parameters())
		p = []
		for i in parameters:
			p+= list(i.data.cpu().numpy().flat)
		print('Number of parameters=',len(p))
		p = None	
		self.model.train()
		print( "Training mode: ", self.model.training)

		print( "Start training")
		########LOSS FUNCTION######
		#loss_fn1 = t.nn.BCELoss(size_average=False)
		#loss_fn1 = AUCLoss()
		loss_fn1 = t.nn.BCEWithLogitsLoss(reduction="sum")
		########OPTIMIZER##########	

		self.learning_rate = learning_rate	
		optimizer = t.optim.Adam(parameters, lr=self.learning_rate, weight_decay=weight_decay)
		scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

		
		########DATALOADER#########
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=None, num_workers=0)
		minLoss = 1000000000
		#######TRAINING ITERATIONS#############
		e = 0
		while e < epochs:			
			errTot = 0
			i = 1
			start = time.time()
			optimizer.zero_grad()
			t1 = time.time()	
			for sample in loader:						
				x, xc, y1, l = sample
				x = x.to(dev)
				xc = xc.to(dev)
				y1 = y1.to(dev)
				l = l.to(dev)
				#print "new sample: ", x.size()
				#print y.size()
				yp, _ = self.model.forward(x, xc, l)
				#print yp.size(), y1.size()
				loss1 = loss_fn1(yp, y1)	
				errTot += loss1.data					
				loss1.backward()
				#print i	
				optimizer.step()
				optimizer.zero_grad()
				t2 = time.time()
				if i % 10 == 0:
					sys.stdout.write("\rbatch: %d/%d (%3.2f%%) %fs" % (i, len(dataset), 100* (i/float(len(dataset))), t2-t1))
				i += batch_size								
			end = time.time()						
			print (" epoch %d, ERRORTOT: %f (%fs)" % (e,errTot, end-start))

			if LOG:
				loss = errTot
				net = self.model
				step = e
				#============ TensorBoard logging ============#
				# (1) Log the scalar values
				info = {'loss': loss}

				for tag, value in info.items():
					logger.scalar_summary(tag, value, step+1)

				# (2) Log values and gradients of the parameters (histogram)
				for tag, value in net.named_parameters():
					tag = tag.replace('.', '/')
					logger.histo_summary(tag, to_np(value), step+1)
					logger.histo_summary(tag+'/grad', to_np(value.grad), step+1)
				#============ TensorBoard logging ============#
			scheduler.step(float(errTot))
			if e % save_model_every == 0 and e > 0:
				#print "Store model ", e
				t.save(self.model, "models/"+self.model.name+".iter_"+str(e)+".t")				
			stdout.flush()
										
			e += 1	
		t.save(self.model, "models/"+self.model.name+".final.t")
	
	def predict(self, X, Xcond, lens, dev, batch_size = 3001, plotGraph=False): # batch_size = 3001 501
		self.model.eval()
		#print "Training mode: ", self.model.training
		if plotGraph:
			from pytorchUtils.torchgraphviz1 import make_dot, make_dot_from_trace
		#print "Predicting..."
		dataset = predictDataset(X, Xcond, lens)
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=None, num_workers=2)
		yp = []	
		ap = []
		first = True
		for sample in loader:
			x, xc, l = sample	
			x = x.to(dev)
			xc = xc.to(dev)
			l = l.to(dev)
			p, a = self.model.forward(x, xc, l)
			pred = t.sigmoid(p)
			#pred = self.model.forward(x, xss, l)
			if first and plotGraph:
				first = False
				print( "printing")
				#print dict(self.model.named_parameters())
				#raw_input()
				make_dot(pred.mean(), params=dict(self.model.named_parameters()))	
			if len(pred.shape)>0:
				yp += pred.tolist()
			else:
				yp +=[float(pred.data)]
			ap += a.tolist()
		return yp, ap

def weighted_binary_cross_entropy(output, target, weights=None):
        
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] * (target * t.log(output)) + \
               weights[0] * ((1 - target) * t.log(1 - output))
    else:
        loss = target * t.log(output) + (1 - target) * t.log(1 - output)

    return t.neg(t.mean(loss))


