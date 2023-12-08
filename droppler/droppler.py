#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  standalone.py
#  
#  Copyright 2018  <orlando.gabriele89@gmail.com>
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
import warnings
import math
import torch
warnings.filterwarnings("ignore")
import argparse,pickle,os
import SA as SA
from NNWrappers import  NNwrapper

import utils
import numpy as np

def run(args):
	args=args[1:]
	pa = argparse.ArgumentParser()
	
	
	
	pa.add_argument('-fasta', '--fasta',
						help='the input FASTA file containing the protein sequences (command line input only)',
						)
	pa.add_argument('-s','--salt',
						help='Ionic strength of the solution (command line input only)',
						default=0.0
						)
	pa.add_argument('-ph','--ph',
						help='pH of the solution (command line input only)',
						default=7.0
						)
	pa.add_argument('-c','--concentration',
						help='protein concentration (uM) (command line input only)',
						default=10
						)
	pa.add_argument('-t','--temperature',
						help='temperature (°C) (command line input only)',
						default=37
						)
	pa.add_argument('-crowd','--crowdingAgent',
						help='1 = presence of crowding agents, 2 = absence of crowding agents (command line input only)',
						default=0
						)
	pa.add_argument('-cuda', '--cuda',
						help='run predictions on GPU. It requires pytorch with CUDA',
						action='store_true',
						default=True)
						
	pa.add_argument('-o', '--outfile',
						help='output file',
						default=False)
	pa.add_argument('-a', '--commandFile',
						help='predict using an input CSV. Provide the path to the CSV file. See the README for its format. ALL THE OTHER CONDITIONS PARAMETERS PROVIDED IN COMMAND LINE WILL BE IGNORED',
						default=False)	
	
													
	results = pa.parse_args(args)

	#CPU = results.cpu
	

	try:
		#model=torch.load("marshalled/model.mtorch")
		if results.cuda:
			device="cuda"
		else:
			device="cpu"
		model = SA.selfatt_RRN(nfea = 20, nCond = 14, hidden_size = 10, num_layers = 2, heads = 10, name = "test").to(device)
		model.load_state_dict(torch.load("marshalled/model.weights"))
		model.eval()
		wrapper = NNwrapper(model)
		scaler = pickle.load(open("marshalled/scaler.pickle","rb"))
		scaler.clip = False # solve a bug in sklearn  as AttributeError
		#print(scaler)
		#asd
	except:
		print("Problems with the serialized model. Please check your pytorch and sklearn versions")
		return
		
	if results.commandFile is False:
		seqs=utils.leggifasta(results.fasta)
		ph = float(results.ph)
		conc = float(results.concentration)
		temp = [0]*10
		temp[math.trunc(int(results.temperature)/10)]=1
		crowd = int(results.crowdingAgent)
		salt = float(results.salt)
		diz={}
		for i in seqs.keys():
			diz[i]=(seqs[i],temp,conc,ph,salt,crowd,"")
			
		X, Xcond, lens, protlist,_= utils.buildVectors(diz)
		
		tmpx = []
		for xa in Xcond:
			tmpx.append(xa.tolist())

		Xcond = scaler.transform(np.array(tmpx))
		
		Xtemp = []
		for i in tmpx:
			Xtemp.append(torch.tensor(i, dtype=torch.float32))
			
		Yp, _ = wrapper.predict(X, Xtemp, lens, dev = device)
		if results.outfile is False:
			print("PROTEIN NAME\tPS PROB")
			for i in range(len(Yp)):
				
				print(protlist[i]," ",round(Yp[i],3))
		else:
			f=open(results.outfile,"w")
			f.write("PROTEIN NAME\tTEMPERATURE\tCONCENTRATION\tPH\tIONIC STRENGTH\tCROWDING AGENT\tLLPS PROBABILITY\n")
			for i in range(len(Yp)):
				#print("PROTEIN NAME\tPS PROB")
				
				f.write(protlist[i]+"\t"+str(results.temperature)+"\t"+str(conc)+"\t"+str(ph)+"\t"+str(salt)+"\t"+str(bool(crowd))+"\t"+str(round(Yp[i],3))+"\n")

	else:
		print("running CSV input. Reading conditions from there...")
		fil=results.commandFile
		diz=utils.parse_csv(fil)

		X, Xcond, lens, protlist,cstring= utils.buildVectors(diz,unique_prots=False)
		
		tmpx = []
		for xa in Xcond:
			tmpx.append(xa.tolist())
		
		Xcond = scaler.transform(np.array(tmpx))
		
		Xtemp = []
		for i in tmpx:
			Xtemp.append(torch.tensor(i, dtype=torch.float32))
			
		Yp, _ = wrapper.predict(X, Xtemp, lens, dev = device)
		if results.outfile is False:
			print("PROTEIN NAME\tPS PROB")
			for i in range(len(Yp)):
				name,seq,temp,conc,ph,salt,crowd = cstring[i]
				print(name," ","Temp = ",temp,"Conc : ",conc," pH : ",ph,"Ionic S : ", salt," Crowd : ",str(bool(crowd)),"LLPS Prob : ", round(Yp[i],3))
		else:
			f=open(results.outfile,"w")
			f.write("PROTEIN NAME\tTEMPERATURE\tCONCENTRATION\tPH\tIONIC STRENGTH\tCROWDING AGENT\tLLPS PROBABILITY\n")
			for i in range(len(Yp)):
				#print("PROTEIN NAME\tPS PROB")
				name,seq,temp,conc,ph,salt,crowd = cstring[i]
				f.write(protlist[i]+"\t"+str(temp)+"\t"+str(conc)+"\t"+str(ph)+"\t"+str(salt)+"\t"+str(bool(crowd))+"\t"+str(round(Yp[i],3))+"\n")
		
	print("Done!\nThe monkeys are listening")
	
if __name__ == '__main__':
    import sys
    sys.exit(run(sys.argv))
