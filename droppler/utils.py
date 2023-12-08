#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  utils.py
#  
#  Copyright 2020 Gabriele Orlando <orlando.gabriele89@gmail.com>
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
import torch,math
from torch.nn.utils.rnn import pad_sequence
import numpy as np
def leggifasta(database): #legge un file fasta e lo converte in un dizionario
		f=open(database)
		uniprot=f.readlines()
		f.close()
		dizio={}
		for i in uniprot:
			#c=re.match(">(\w+)",i)  4
		
			if i[0]=='>':
					if '|' in i:
						uniprotid=i.strip('>\n').split('|')[1]
					else:
						uniprotid=i.strip('>\n').split(' ')[0]
					dizio[uniprotid]=''
			else:
				dizio[uniprotid]=dizio[uniprotid]+i.strip('\n')
		return dizio
def getEmbeddingValues(s):
	r = []
	listAA=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
	for i in s:
		if i not in listAA:
			r.append(0)
		else:
			r.append(listAA.index(i)+1)	
	return r
def buildVectors(seqs,unique_prots=True):	#db = {uid:(seq,label)}
	X = []
	Xcond = []
	Y = []
	corresp = []
	lens=[]
	protlist=[]
	cstring=[]
	for u in seqs.keys():
		seq,temp,conc,ph,salt,crowd,cond_string  = seqs[u]
		if unique_prots:
			protlist+=[u]
			
		else:
			protlist+=[cond_string[0]]
		cstring+=[cond_string]
		assert sum(temp) == 1
		embe=torch.tensor(getEmbeddingValues(seq))
		X+=[embe]

		Xcond.append(temp+[conc, crowd, salt, ph])
		lens+=[len(embe)]
		corresp.append(u)
		
	Xcond=np.array(Xcond)

	tmp = []

	
	X = pad_sequence(X,padding_value=0,batch_first=True)

	return X, Xcond, lens, protlist,cstring

def parse_csv(fil):
	diz={}
	for i in open(fil).readlines():
	
		name,seq,temp,conc,ph,salt,crowd= i.split()
		
		t = [0]*10
		t[math.trunc(int(temp)/10)]=1
		
		diz[(name,temp,conc,ph,salt,crowd)]=(seq,t,float(conc),float(ph),float(salt),int(crowd),(name,seq,temp,conc,ph,salt,crowd))
	return diz
	
