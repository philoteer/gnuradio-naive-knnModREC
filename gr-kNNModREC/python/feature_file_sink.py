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

import numpy
from gnuradio import gr
import pmt
import csv
import time
import os.path

#stores feature data to a dataset file.
#in: two float vectors (PSD estimates : min, max), a message port (for signal mapping data).
#out: none (sink)

class feature_file_sink(gr.sync_block):
	"""
	docstring for block feature_file_sink
	"""
	#constructor
	def __init__(self, vec_len1, vec_len2,file_path, sig_no):
		gr.sync_block.__init__(self,
			name="feature_file_sink",
			in_sig=[(numpy.float32,vec_len1),(numpy.float32,vec_len2)],
			out_sig=None)
		
		#init the message port, handler
		self.message_port_register_in(pmt.intern('map'))
		self.set_msg_handler(pmt.intern("map"), self.map_handler)
		
		#init vars
		self.vec_len1 = vec_len1
		self.vec_len2 = vec_len2
		self.file_path = file_path		
		self.freq_list = []
		self.bw_list = []
		self.sig_no = sig_no
		self.in0_avg = numpy.zeros(vec_len1)
		self.in1_avg = numpy.zeros(vec_len2)
		self.signal_name = ""

		#if file exists, append. if not, create new.
		if os.path.isfile(file_path):
			self.csv_out = open(file_path,'ab')		#append
		else:
			self.csv_out = open (file_path,'wb')	#overwrite
			self.csv_out.write("frequency,")
			for i in range (0,vec_len1):
				self.csv_out.write("v1_"+str(i)+",")
			for i in range (0,vec_len2):
				self.csv_out.write("v2_"+str(i)+",")
			self.csv_out.write("class\n")

	#message handler (gets and stores signal mapping data)
	def map_handler(self,msg):
		content = pmt.to_python(msg)	#type:float
		length = pmt.length(msg)
		
		#list of signals (their freqs and bandwidths)
		self.freq_list = []
		self.bw_list = []
		for i in range (0, length):
			self.freq_list.append(content[i][0])
			self.bw_list.append(content[i][1])

	#capture event!
	def capture_handler(self):
		self.csv_out.write(str(self.bw_list[self.sig_no]) + "," + ','.join([str(x) for x in self.in0_avg]) + ',' + ','.join([str(x) for x in self.in1_avg]) + "," + self.signal_name +"\n")
		print ""
		print "Data captured:" + str(self.freq_list[self.sig_no]) + "/" + str(self.bw_list[self.sig_no]) + "," + self.signal_name		
		print ""

	#main work thread.
	def work(self, input_items, output_items):
		#get data stream
		in0 = input_items[0]
		in1 = input_items[1]
		
		#Update feature data vectors.
		if len(in0 >= 1):
			self.in0_avg = numpy.mean(in0,axis=0)
			self.in1_avg = numpy.mean(in1,axis=0)
		
		#Empty the data buffer
		self.consume(0, len(in0))
		self.consume(1, len(in1))
		return len(input_items[0])

	#Capture Button callback
	def input_callback(self, capture_button, signal_name):
		self.signal_name = signal_name
		if capture_button != 0:
			self.capture_handler()
