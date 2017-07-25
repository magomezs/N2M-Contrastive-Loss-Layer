import caffe
import numpy as np
import os
import sys

class N2MCLoss(caffe.Layer):
  def setup(self, bottom, top):
    	if len(bottom) != 3:
       		raise Exception('Must have exactly three inputs: two arrays of descriptors (A and B) and one array of labels')
    	if len(top) != 1:
       		raise Exception('Must have exactly one output: the loss')
 

  def reshape(self,bottom,top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            	raise Exception("Inputs must have the same dimension.")
        # check number of labels 
        if bottom[2].num != bottom[0].num:
            	raise Exception("Incoherence in the number of labels.")
        
	# loss output is scalar
        top[0].reshape(1)
       
	#Margins
 	self.m1 = 0.3
        self.m2 = 0.7

        #New varables
	self.diff = np.zeros((bottom[0].num, bottom[0].channels), dtype=np.float32)  	#Descripstors difference has descriptors dimensions
        self.dist = np.zeros((bottom[0].num, 1), dtype=np.float32)			#Euclidean Distance is the connection function
	self.dist_norm = np.zeros((bottom[0].num, 1), dtype=np.float32)			#Normalised connection function
       
        #Auxiliar arrays to compute loss
        self.Y = np.zeros((bottom[0].num, 1), dtype=np.float32)			#Array of labels
        self.zeros = np.zeros((bottom[0].num, 1), dtype=np.float32)		#Array of zeros with labels array dimensions
        self.ones = np.ones((bottom[0].num, 1), dtype=np.float32)		#Array of ones with labels array dimensions
	self.losses= np.zeros((bottom[0].num, 1), dtype=np.float32)		#Array of losses for every sample of a batch

	#Variables for backwards propagation
        self.mask1 = np.zeros((bottom[0].num, 1), dtype=np.float32)		#All of them have labels array dimension
        self.mask2 = np.zeros((bottom[0].num, 1), dtype=np.float32)
        self.factors = np.zeros((bottom[0].num, 1), dtype=np.float32)
        self.weights = np.zeros((bottom[0].num, 1), dtype=np.float32)
		
		
  def forward(self, bottom, top):	   
	self.Y[...,0] = bottom[2].data
        self.diff = bottom[0].data - bottom[1].data
        self.dist[..., 0] = np.sqrt(np.sum(self.diff**2, axis=1))   
	self.dist_norm = 2.0*((1/(self.ones + np.exp((-1.0)*self.dist)))-(0.5*self.ones)) 
	self.losses = (self.Y * np.max([self.zeros, (self.dist_norm - (self.m1*self.ones))], axis=0) ) + ((self.ones - self.Y )* np.max([self.zeros, ((self.m2*self.ones) - self.dist_norm)], axis=0) )
        top[0].data[0] = np.sum(self.losses)/ (2.0 * bottom[0].num)

    
  def backward(self, top, propagate_down, bottom):   
	self.Y[...,0] = bottom[2].data
	self.mask1 = np.where((self.dist_norm - (self.m1 * self.ones)) > 0.0, 1.0, 0.0)
	self.mask2 = np.where(((self.m2 * self.ones) - self.dist_norm) > 0.0, 1.0, 0.0)
        for i, sign in enumerate([ +1, -1 ]):
        	if propagate_down[i]:
         		self.factors = np.where(self.Y>0, +1.0, -1.0) * sign * top[0].diff[0]/ bottom[i].num
			self.weights=((self.ones-self.Y)*self.mask2 + self.Y*self.mask1)*self.factors
			bottom[i].diff[...] = np.array([self.weights]).T * self.diff
 

