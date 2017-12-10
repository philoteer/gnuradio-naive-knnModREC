#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Copyright 2017 <+YOU OR YOUR COMPANY+>.
# 
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
# 

import numpy as np
from gnuradio import gr
import pandas as pd
import pmt
from sklearn.neighbors import KNeighborsClassifier

class kNN_ModRec(gr.sync_block):
	"""
	docstring for block kNN-ModRec
	"""
	def __init__(self,vec_len1, vec_len2, file_path, sig_no, metric_type, n):
		gr.sync_block.__init__(self,
			name="kNN_ModRec",
			in_sig=[(np.float32,vec_len1),(np.float32,vec_len2)],
			out_sig=None)

		self.message_port_register_in(pmt.intern('map'))
		self.set_msg_handler(pmt.intern("map"), self.map_handler)
		
		self.vec_len1 = vec_len1
		self.vec_len2 = vec_len2
		self.file_path = file_path
		self.sig_no = sig_no
		self.freq_list = []
		self.bw_list = []
		
		self.df = pd.read_csv(file_path, header = 0)
		self.y = self.df['class']
		self.x = self.df.drop(['class'],1)
		
		self.knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric=metric_type,
			metric_params=None, n_jobs=1, n_neighbors=n, p=2, weights='uniform')
			
		self.knn.fit(self.x,self.y)
		
		self.message_port_register_out(pmt.intern('classification'))
		
	def map_handler(self,msg):
		content = pmt.to_python(msg)	#type:float
		length = pmt.length(msg)
		
		self.freq_list = []
		self.bw_list = []
		for i in range (0, length):
			self.freq_list.append(content[i][0])
			self.bw_list.append(content[i][1])
			

	def callback(self, metric_type, n):
		self.knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric=metric_type,
			metric_params=None, n_jobs=1, n_neighbors=n, p=2, weights='uniform')
			
		self.knn.fit(self.x,self.y)
		print "setting update:alg="+metric_type+"/n="+str(n)
				

	def work(self, input_items, output_items):
		in0 = input_items[0]
		in1 = input_items[1]
		
		if len(in0) >= 1:
			x = [self.bw_list[self.sig_no]]
			x.extend(in0[0])
			x.extend(in1[0])
			x_tmp = []

			x_tmp.append(x)
			a = self.knn.predict(x_tmp)

			#print "Signal #"+ str(self.sig_no)+" :" +a[0]
			
			pmt_signal = pmt.to_pmt(("signal",self.sig_no))
			pmt_class = pmt.to_pmt((a[0],0))
			pmt_tuple = pmt.make_tuple(pmt_signal,pmt_class)			

			self.message_port_pub(pmt.intern("classification"),pmt_tuple)

		self.consume(0, len(in0))
		self.consume(1, len(in1))
		return len(input_items[0])

