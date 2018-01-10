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

#take average of a vector.
#in: a float32 vector (len = vec_len)
#out: a float32 vector (len = vec_len)

class vector_avg(gr.basic_block):
	"""
	docstring for block vector_avg
	"""
	#constructor
	def __init__(self, vec_len, avg_len):
		gr.basic_block.__init__(self,
			name="vector_avg",
			in_sig=[(numpy.float32,vec_len)],
			out_sig=[(numpy.float32,vec_len)])
			
		#store args
		self.avg_len = avg_len
		self.vec_len = vec_len
		self.cnt = 0
		self.result = numpy.zeros(vec_len)
			
	#worker
	def general_work(self, input_items, output_items):

		in0 = input_items[0]
		out = output_items[0]
		
		#length of data to process
		proc_len = len(in0) if len(in0) + self.cnt <= self.avg_len else self.avg_len - self.cnt
		self.cnt += proc_len
		
		#sum vectors that are to be processed (divided by the avg len later).
		self.result = self.result + numpy.sum(in0[0:proc_len],axis=0)
		
		#consume
		self.consume(0, proc_len)
		
		#if we need to output a vector, finish averaging and write to output buffer.
		if self.cnt >= self.avg_len and len(out) >= 1:
			out[0] =  (1.0/self.avg_len) * self.result
			self.result = numpy.zeros(self.vec_len)
			self.cnt = 0
			return 1
		else:
			return 0
