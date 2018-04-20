import caffe
import numpy as np
import os
import io
import sys

class N2MCLoss(caffe.Layer):
  def setup(self, bottom, top):
    	if len(bottom) != 4:
       		raise Exception('Must have exactly 4 inputs: two arrays of descriptors (A and B), distances,  and one array of labels')
    	if len(top) != 1:
       		raise Exception('Must have exactly one output: the loss')
	#Parameters reading
	params = eval(self.param_str)
       	self.m1 = params["m1"] #A reasonable value for margin 1 is 0.3
	self.m2 = params["m2"] #A reasonable value for margin 2 is 0.7
		
  def reshape(self,bottom,top):
	#check inputs sizes
        if bottom[0].count != bottom[1].count:
			raise Exception('Inputs (dist_a and dist_b) must have the same dimension')
        if bottom[2].count != bottom[3].count:
			raise Exception('Inputs (M_dist and labels) must have the same dimension')
        
	# loss output is scalar
        top[0].reshape(1)
       
	#New varables
	self.dist = np.zeros((bottom[0].num, 1), dtype=np.float32)	        #connection function
	self.dist_norm = np.zeros((bottom[0].num, 1), dtype=np.float32)		#Normalised connection function
       
        #Auxiliar arrays to compute loss
        self.Y = np.zeros((bottom[0].num, 1), dtype=np.float32)			#Array of labels
        self.zeros = np.zeros((bottom[0].num, 1), dtype=np.float32)		#Array of zeros with labels array dimensions
        self.ones = np.ones((bottom[0].num, 1), dtype=np.float32)		#Array of ones with labels array dimensions
	self.losses= np.zeros((bottom[0].num, 1), dtype=np.float32)		#Array of losses for every sample of a batch

	#Variables for backwards propagation
        self.mask1 = np.zeros((bottom[0].num, 1), dtype=np.float32)		#All of them have the same dimension as labels array 
	self.mask2 = np.zeros((bottom[0].num, 1), dtype=np.float32)
        self.factors = np.zeros((bottom[0].num, 1), dtype=np.float32)
        self.weights = np.zeros((bottom[0].num, 1), dtype=np.float32)
		
		
  def forward(self, bottom, top):	   
	self.Y[...,0] = bottom[3].data
        self.diff = bottom[0].data - bottom[1].data
        self.dist[..., 0] = bottom[2].data  
	#print self.dist
	self.dist_norm = 2.0*((1/(self.ones + np.exp((-1.0)*self.dist)))-(0.5*self.ones)) 
	self.losses = (self.Y * np.max([self.zeros, (self.dist_norm - (self.m1*self.ones))], axis=0) ) + ((self.ones - self.Y )* np.max([self.zeros, ((self.m2*self.ones) - self.dist_norm)], axis=0) )
        top[0].data[0] = np.sum(self.losses)/ (2.0 * bottom[0].num)

    
  def backward(self, top, propagate_down, bottom):   
	self.Y[...,0] = bottom[3].data
	self.mask1 = np.where((self.dist_norm - (self.m1 * self.ones)) > 0.0, 1.0, 0.0)
	self.mask2 = np.where(((self.m2 * self.ones) - self.dist_norm) > 0.0, 1.0, 0.0)
        for i, sign in enumerate([ +1, -1 ]):
        	if propagate_down[i]:
         		self.factors = np.where(self.Y>0, +1.0, -1.0) * sign * top[0].diff[0]/ bottom[i].num
			self.weights=((self.ones-self.Y)*self.mask2 + self.Y*self.mask1)*self.factors
			bottom[i].diff[...] = np.array([self.weights]).T * self.diff
