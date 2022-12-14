
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch as t
from pysot.core.config import cfg
from pysot.utils.bbox import IoU



class AnchorTarget():
    def __init__(self):

        return
    def select(self,position, keep_num=16):
        num = position[0].shape[0]
        if num <= keep_num:
            return position, num
        slt = np.arange(num)
        np.random.shuffle(slt)
        slt = slt[:keep_num]
        return tuple(p[slt] for p in position), keep_num
    

    def get(self,bbox,size):
           
        labelcls1=np.zeros((1,size,size))-1

        pre=(16*(np.linspace(0,size-1,size))+63).reshape(-1,1)-cfg.TRAIN.SEARCH_SIZE//2
        pr=np.zeros((size**2,2))
        pr[:,0]=np.maximum(0,np.tile(pre,(size)).T.reshape(-1)+cfg.TRAIN.SEARCH_SIZE//2)
        pr[:,1]=np.maximum(0,np.tile(pre,(size)).reshape(-1)+cfg.TRAIN.SEARCH_SIZE//2)
    
        labelxff=np.zeros((4, size, size), dtype=np.float32)
        
        labelcls2=np.zeros((1,size,size))
        weightxff=np.zeros((1,size,size))

        
        target=np.array([bbox.x1,bbox.y1,bbox.x2,bbox.y2])
        
        index2=np.int32(np.minimum(size-1,np.maximum(0,(target-63)/16)))
        w=int(index2[2]-index2[0]+1)
        h=int(index2[3]-index2[1]+1)
        
        range=cfg.TRAIN.LABEL_RANGE
        for ii in np.arange(0,size):
            for jj in np.arange(0,size):
                 weightxff[0,ii,jj]=(((ii-(index2[1]+index2[3])/2)*range)**2+((jj-(index2[0]+index2[2])/2)*range)**2)
        
        
        se=weightxff[np.where(weightxff<((w//2+h//2)*range/1.5)**2)]
        
        weightxff[np.where(weightxff<((w//2+h//2)*range/1.5)**2)]=1-((se-se.min())/(se.max()-se.min()+1e-4))
        
        weightxff[np.where(weightxff>((w//2+h//2)*range/1.5)**2)]=0
        
        pos=np.where(weightxff.squeeze()>0.8)
        num=len(pos[0])
        pos = self.select(pos, num//4)
        weightxff[:,pos[0][0],pos[0][1]] = 1.5            

        

        index=np.int32(np.minimum(size-1,np.maximum(0,(target-63)/16)))
        w=int(index[2]-index[0]+1)
        h=int(index[3]-index[1]+1)



        for ii in np.arange(0,size):
            for jj in np.arange(0,size):
                  labelcls2[0,ii,jj]=(((ii-(index[1]+index[3])/2)*range)**2+((jj-(index[0]+index[2])/2)*range)**2)
                 
                 
        see=labelcls2[np.where(labelcls2<((w//2+h//2)*range/1.2)**2)]
        
        labelcls2[np.where(labelcls2<((w//2+h//2)*range/1.2)**2)]=1-((see-see.min())/(see.max()-see.min()+1e-4))
        weightcls3=np.zeros((1,size,size))
        weightcls3[np.where(labelcls2<((w//2+h//2)*range/1.2)**2)]=1
        labelcls2=labelcls2*weightcls3


        def con(x):
           return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))  
        def dcon(x):
           return (np.log(1+x)-np.log(1-x))/2 

        labelxff[0,:,:]=(pr[:,0]-target[0]).reshape(cfg.TRAIN.OUTPUTFEATURE_SIZE,cfg.TRAIN.OUTPUTFEATURE_SIZE)
        labelxff[1,:,:]=(target[2]-pr[:,0]).reshape(cfg.TRAIN.OUTPUTFEATURE_SIZE,cfg.TRAIN.OUTPUTFEATURE_SIZE)
        labelxff[2,:,:]=(pr[:,1]-target[1]).reshape(cfg.TRAIN.OUTPUTFEATURE_SIZE,cfg.TRAIN.OUTPUTFEATURE_SIZE)
        labelxff[3,:,:]=(target[3]-pr[:,1]).reshape(cfg.TRAIN.OUTPUTFEATURE_SIZE,cfg.TRAIN.OUTPUTFEATURE_SIZE)
        labelxff=con(labelxff/(cfg.TRAIN.SEARCH_SIZE//2))

           
        

        labelcls1[0,index[1]-h//4:index[3]+1+h//4,index[0]-w//4:index[2]+1+w//4]=-2
        labelcls1[0,index[1]+h//4:index[3]+1-h//4,index[0]+w//4:index[2]+1-w//4]=1
        
        neg2=np.where(labelcls1.squeeze()==-1)
        neg2 = self.select(neg2, int(len(np.where(labelcls1==1)[0])*2.5))
        labelcls1[:,neg2[0][0],neg2[0][1]] = 0
        
     
        return  labelcls1,labelxff,labelcls2,weightxff


