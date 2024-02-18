# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:54:33 2022

@author: wingh
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math
from copy import copy, deepcopy

#-----------    
# function for  encoding image
#-----------

def BTCencode_x(fname_in,fname_out):
    img = mpimg.imread(fname_in)# img is a np.array
    #implot = plt.imshow(img, cmap='gray')# show image
    print(img)
    print(type(img))
    #-----------
    X = np.array(img)*255
    
    rows_of_blocks = int(X.shape[0]/3)
    cols_of_blocks = int(X.shape[1]/3)
    
    
    blk_m = np.zeros([rows_of_blocks,cols_of_blocks],dtype="uint8") # reserve a blk_m plane
    blk_std = np.zeros([rows_of_blocks,cols_of_blocks],dtype="uint8") # reserve a blk_std plane
    blk_index = np.zeros([rows_of_blocks,cols_of_blocks],dtype="uint8") # reserve a blk_index plane
    blk_all1 = np.zeros([rows_of_blocks,cols_of_blocks],dtype="uint8") # reserve a all result plane for 8-bits
    blk_all2 = np.zeros([rows_of_blocks,cols_of_blocks],dtype="uint8") # reserve a all result plane for 8-bits
    
    
    for i in range(0,X.shape[0]-3,3):
        for j in range(0,X.shape[1]-3,3):
            block = X[i:i+3,j:j+3] # get an 3x3 block
            #print(i,j)
            #print(block)
            
            miu = np.mean(block) # get the mean of the block
            block_std = np.std(block) # get the std of the block
            newU = min(255,2*((miu/2)//1)) # get the u' of the block
            newSTD = min(127,4*((block_std/4)//1)) # get the STD' of the block
            g0 = max(0,newU-newSTD)
            g1 = min(255,newU + newSTD)
            codebook = np.zeros((16,3,3))
            codebook[0]=[[g0,g0,g1],[g0,g1,g1],[g1,g1,g1]]   # set the pattern of codebook 0-16
            codebook[1]=[[g1,g0,g0],[g1,g1,g0],[g1,g1,g1]]
            codebook[2]=[[g1,g1,g1],[g1,g1,g0],[g1,g0,g0]]
            codebook[3]=[[g1,g1,g1],[g0,g1,g1],[g0,g0,g1]]
            codebook[4]=[[g0,g0,g0],[g1,g1,g1],[g1,g1,g1]]
            codebook[5]=[[g1,g1,g1],[g1,g1,g1],[g0,g0,g0]]
            codebook[6]=[[g1,g1,g0],[g1,g1,g0],[g1,g1,g0]]
            codebook[7]=[[g0,g1,g1],[g0,g1,g1],[g0,g1,g1]]
            codebook[8]=[[g1,g1,g0],[g1,g0,g0],[g0,g0,g0]]
            codebook[9]=[[g0,g1,g1],[g0,g0,g1],[g0,g0,g0]]
            codebook[10]=[[g0,g0,g0],[g0,g0,g1],[g0,g1,g1]]
            codebook[11]=[[g0,g0,g0],[g1,g0,g0],[g1,g1,g0]]
            codebook[12]=[[g1,g1,g1],[g0,g0,g0],[g0,g0,g0]]
            codebook[13]=[[g0,g0,g0],[g0,g0,g0],[g1,g1,g1]]
            codebook[14]=[[g0,g0,g1],[g0,g0,g1],[g0,g0,g1]]
            codebook[15]=[[g1,g0,g0],[g1,g0,g0],[g1,g0,g0]]

            w = 15
            min_distance=99999
            index = 0
            while(w>=0):                                      # find the minimum distance and get the indexed to be saved
                distance = np.linalg.norm(block-codebook[w])   
                if(distance<=min_distance):
                    min_distance=distance
                    index = w
                w -= 1
                
            
            blk_m[int(i/3),int(j/3)] = np.uint8(newU//1)          # store mean 
            blk_std[int(i/3),int(j/3)] = np.uint8(newSTD//1)      # store STD 
            blk_index[int(i/3),int(j/3)] = np.uint8(index//1)     # store index
            #print(newU)
            #print(newSTD)
            #print(index)
            if(newSTD/4>16):
                blk_all1[int(i/3),int(j/3)] = np.uint8(newU)+np.uint8(1)        # store the first 7 bits and most significant nit of STD
                blk_all2[int(i/3),int(j/3)] = np.uint8(newSTD/4-16)*(2**4)+np.uint8(index) # store the rest 4 bits of STD and 4 bits index
            else:
                blk_all1[int(i/3),int(j/3)] = np.uint8(newU)+np.uint8(0)
                blk_all2[int(i/3),int(j/3)] = np.uint8(newSTD/4)*(2**4)+np.uint8(index)
    
    file = open(fname_out,"wb")
    header = np.zeros([8],dtype=('uint8'))   # store header file
    header[0] = np.uint8(8)
    header[1] = np.uint8(3)
    header[2] = np.uint8(min(255,X.shape[1]//3))
    header[3] = np.uint8(max(0,X.shape[1]//3-255))   
    header[4] = np.uint8(X.shape[1]%3)
    header[5] = np.uint8(min(255,X.shape[0]//3))
    header[6] = np.uint8(max(0,X.shape[0]//3-255))
    header[7] = np.uint8(X.shape[0]%3)
    
    for byte in header:
        file.write(byte)
    
    for i in range(0,rows_of_blocks):         # store information of each block
        for j in range(0,cols_of_blocks):
            file.write(blk_all1[i,j])
            file.write(blk_all2[i,j])
    file.close()
    
    
    return ({'blk_mean':blk_m,'blk_SD':blk_std,'blk_index':blk_index})
            
def int2bin(x):
    return(1*((x%np.array([256,128,64,32,16,8,4,2])-x%np.array([128,64,32,16,8,4,2,1]))>0))
        
def BTCdecode_x(fname_in,fname_out):
    #------read a btc-encoded  image from a file
    file = open(fname_in,"rb")
    
    #------read the header
    header_len = file.read(1)[0] 
    block_size = file.read(1)[0]
    no_of_block_rows = file.read(1)[0]+file.read(1)[0]*255
    no_of_skipped_rows = file.read(1)[0]
    no_of_block_cols = file.read(1)[0]+file.read(1)[0]*255
    no_of_skipped_cols = file.read(1)[0]
    
    file.read(header_len-8)
    #print(no_of_block_rows*block_size+no_of_skipped_rows)
    #print(no_of_block_cols*block_size+no_of_skipped_cols)
    OImg = np.zeros((no_of_block_cols*block_size+no_of_skipped_cols,no_of_block_rows*block_size+no_of_skipped_rows))  #create array to store the rebuild image
    
    Out = np.zeros([3,3],dtype = 'uint8')

    for i in range(0,no_of_block_rows*block_size,3):      #rebuild most of the image block by block
        for j in range(0,no_of_block_cols*block_size,3):         
            blk_h=np.uint8(file.read(1)[0]) #read first 8-bits
            blk_l=np.uint8(file.read(1)[0]) #read last 8-bits
            blk_re_newU = 2*(np.uint8(blk_h/2))
            #print(blk_newU)
            blk_re_std = 4*(np.uint8((blk_h%2)*(2**4)+blk_l/(2**4)))
            #print(blk_re_std)
            blk_re_index = np.uint8(blk_l%2**4)
            #print(blk_re_index)
            
            g0 = max(0,blk_re_newU-blk_re_std)
            
            g1 = min(255,blk_re_newU+blk_re_std)
            codebook = np.zeros((16,3,3))
            codebook[0]=[[g0,g0,g1],[g0,g1,g1],[g1,g1,g1]]
            codebook[1]=[[g1,g0,g0],[g1,g1,g0],[g1,g1,g1]]
            codebook[2]=[[g1,g1,g1],[g1,g1,g0],[g1,g0,g0]]
            codebook[3]=[[g1,g1,g1],[g0,g1,g1],[g0,g0,g1]]
            codebook[4]=[[g0,g0,g0],[g1,g1,g1],[g1,g1,g1]]
            codebook[5]=[[g1,g1,g1],[g1,g1,g1],[g0,g0,g0]]
            codebook[6]=[[g1,g1,g0],[g1,g1,g0],[g1,g1,g0]]
            codebook[7]=[[g0,g1,g1],[g0,g1,g1],[g0,g1,g1]]
            codebook[8]=[[g1,g1,g0],[g1,g0,g0],[g0,g0,g0]]
            codebook[9]=[[g0,g1,g1],[g0,g0,g1],[g0,g0,g0]]
            codebook[10]=[[g0,g0,g0],[g0,g0,g1],[g0,g1,g1]]
            codebook[11]=[[g0,g0,g0],[g1,g0,g0],[g1,g1,g0]]
            codebook[12]=[[g1,g1,g1],[g0,g0,g0],[g0,g0,g0]]
            codebook[13]=[[g0,g0,g0],[g0,g0,g0],[g1,g1,g1]]
            codebook[14]=[[g0,g0,g1],[g0,g0,g1],[g0,g0,g1]]
            codebook[15]=[[g1,g0,g0],[g1,g0,g0],[g1,g0,g0]]
            
            OImg[i:i+3,j:j+3] = np.copy(codebook[blk_re_index])

            #print("rebuild")
            #print(OImg[i:i+3,j:j+3])
    
    #print(no_of_block_cols*block_size)
    #print(no_of_block_cols*block_size+no_of_skipped_cols)
    
    
    for i in range(0,OImg.shape[0]):       # rebuild rest of the block 
        for j in range(no_of_block_cols*block_size,no_of_block_cols*block_size+OImg.shape[1]%3):
            OImg[i,j]=OImg[i,j-1]
            
    for i in range(no_of_block_rows*block_size,no_of_block_rows*block_size+OImg.shape[0]%3):   # rebuild rest of the block 
        for j in range(0,no_of_block_cols*block_size+OImg.shape[1]%3):
            OImg[i,j]=OImg[i-1,j]
    file.close()
    #print(OImg.size)        
    #myaxis1[1].imshow(OImg,cmap='gray',vmin=(0),vmax=(255))
    imgplot = plt.imshow(OImg,cmap='gray',vmin=(0),vmax=(255))
    plt.imsave(fname_out, OImg, cmap='gray',vmin=(0),vmax=(255))
    
    
    #img = mpimg.imread('myTimg.png')
    #X = np.array(img)*255
    #print(X.size)
    
    
    
    return OImg
    
 #----------
 #main programme 
 #----------
 
try:
    print("Please input an existing file with proper file format such as .png")
    name = input("input file name:")
    fname_in = name
    n = name[0:-4]
    fname_out = n+'_encoded.bvqc3'
    result = BTCencode_x(fname_in,fname_out)
    #print(result['blk_mean'][0,1])
    
    img = mpimg.imread(name) 
    origin = np.array(img)*255 
    L1 = origin.shape[0]
    L2 = origin.shape[1]
    print("origin:")
    print(origin)
    #print(result)
    #BTCencode_x(fname_in,fname_out)
    fname_in = n+'_encoded.bvqc3'
    fname_out = n+'-bvqc3-R.png'
    y = BTCdecode_x(fname_in,fname_out)
    
    img2 = mpimg.imread(fname_out)
    re = np.array(img2)*255
    print("re:")
    print(re[:,:,0])
    tempR = 0
    for i in range(0,L1):
        for j in range(0,L2):
            tempR+=(origin[i,j]-re[i,j,0])**2
            
    mse = 1/(L1*L2)*tempR  #calculate mse
    
    ppsnr = 10*np.log10((255**2)/mse) #calculate ppsnr
    print("The mse is:")
    print(mse)
    print("The ppsnr is:")
    print(ppsnr)
except (OSError) as e:
    print("Error,file not found or can not open file")           
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            