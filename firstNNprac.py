# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 14:18:34 2020

@author: 88697
"""

import numpy as np
import matplotlib.pyplot as plt

input_data = np.arange(0,2*np.pi,0.1)
correct_data = np.cos(input_data)
input_data = (input_data-np.pi)/np.pi
n_data = len(input_data)

n_in =1
n_middle =3
n_out = 1

wb_width =0.01
eta =0.1
epoch=2001
interval = 200

class MiddleLayer:
    def __init__(self,n_upper,n):
        self.w1 =wb_width*np.random.randn(n_upper,n)
        self.b1 =wb_width*np.random.randn(n)
        
    def forward(self,x):
        self.x =x
        u1 =np.dot(x,self.w1)+self.b1
        self.y1 =1/(1+np.exp(-u1))
        
    def backward(self,grad_y1):
        delta1 =grad_y1*(1-self.y1)*self.y1
        self.grad_w1 =np.dot(self.x.T,delta1)
        self.grad_b1 =np.sum(delta1,axis =0)
    
    def update(self,eta):
        self.w1 =self.w1-eta*self.grad_w1
        self.b1 =self.b1-eta*self.grad_b1
        
class OutputLayer:
    def __init__(self,n_upper,n):
        self.w2=wb_width*np.random.randn(n_upper,n)
        self.b2=wb_width*np.random.randn(n)

    def forward(self,y1):
        self.y1=y1
        u2=np.dot(y1,self.w2)+self.b2
        self.y2=u2

    def backward(self,t):
        delta2 =self.y2-t
        self.grad_w2 =np.dot(self.y1.T,delta2)
        self.grad_b2 =np.sum(delta2,axis=0)
        self.grad_y1 =np.dot(delta2,self.w2.T)
        
    def update(self,eta):
        self.w2=self.w2-eta*self.grad_w2
        self.b2=self.b2-eta*self.grad_b2
        
middle_layer =MiddleLayer(1,3)
output_layer =OutputLayer(3,1)

for i in range(epoch):
    
    index_random =np.arange(n_data)
    np.random.shuffle(index_random)
    
    total_error =0
    plot_x =[]
    plot_y2 =[]
    
    for idx in index_random:
        
        x =input_data[idx:idx+1]
        t =correct_data[idx:idx+1]
        
        middle_layer.forward(x.reshape(1,1))
        output_layer.forward(middle_layer.y1)
        
        output_layer.backward(t)
        middle_layer.backward(output_layer.grad_y1)
        
        middle_layer.update(eta)
        output_layer.update(eta)
        
        if i%interval==0:
            y2 =output_layer.y2.reshape(-1)
            total_error += 1.0/2.0*np.sum(np.square(y2-t))
            plot_x.append(x)
            plot_y2.append(y2)
            
    if i%interval==0:
        plt.plot(input_data, correct_data, linestyle="dashed")
        plt.scatter(plot_x,plot_y2, marker="+")
        plt.show()
        print("Epoch:" +str(i)+"/"+str(epoch),"Error:"+str(total_error/n_data))
        
        
        