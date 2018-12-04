import os, copy
from multiprocessing import Pool

import numpy as np
import pandas as pd
import datetime as dt
from statistics import mean

import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torch.autograd import Variable
from torch.utils.data.sampler import SequentialSampler

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from KNet_utils import invDict, matrixPlot
from KNet_Loader import ImageReader

PHASE = ['train','val']
AWNS = ['AWNED','AWNLESS ']
AWNS2 = ['AWNED','AWNLESS']
# normalization paras
RGBmean, RGBstdv = [0.429, 0.495, 0.259], [0.218, 0.224, 0.171]  

def checkfile(path):
    try:
        img = default_loader(path)
    except OSError:
        if os.path.exists(path):
            os.remove(path)
        print('OSError', end ='\r')
    return

class csv2pred:
    ##################################################
    # initialization
    ##################################################
    def __init__(self,src,dst,csvfiles,model_path):
        self.src = src
        self.dst = dst
        self.csvs = csvfiles# a list a csv files
        self.model_path = model_path
        
        self.plot2awns = {}# plot_id -> awns(str)        
        self.previewcsvs()
        
        self.plot2date2acc = {p:{} for p in self.plot2awns}# plot_id -> date -> acc
        self.plot2date2num = {p:{} for p in self.plot2awns}# plot_id -> date -> num

        if not os.path.exists(self.dst): os.makedirs(self.dst)
        if not os.path.exists(self.dst + 'csvs/'): os.makedirs(self.dst + 'csvs/')

        self.acc = [0,0]
        self.cfmat = np.zeros(( len(AWNS), len(AWNS) ))
        
    ##################################################
    # step 0: preview csvs
    ##################################################   
    def previewcsvs(self):
        for file in self.csvs:
            self.df = pd.read_csv(file)
            self.viewcsv(file)
            # self.precheck()   ############ only need for the first time
    
    def viewcsv(self, file):
        # extract info
        df_Plots = pd.Series.to_dict(self.df['plot_id'])          # dict type
        df_AWNS  = pd.Series.to_dict(self.df['AWNS'])             # dict type

        for i, a in df_AWNS.items():
            p = df_Plots[i][6:]
            if p in self.plot2awns:
                if self.plot2awns[p] != a: print('wrong plot info')
            else:
                self.plot2awns[p] = a

    def precheck(self):
        # extract info
        df_Paths = pd.Series.to_dict(self.df['image_file_name'])  # dict type
        
        # check broken images
        path = [self.src+p for idx, p in df_Paths.items()]
        Pool(32).map(checkfile,path)   
        
    ##################################################
    # step 1: setModel
    ##################################################   
    def setModel(self):
        # load model
        self.model = torch.load(self.model_path+'model.pth').module
        self.transforms = transforms.Compose([transforms.Scale(224*4),
                                              transforms.CenterCrop(224*3),
                                              transforms.ToTensor(),
                                              transforms.Normalize(RGBmean, RGBstdv)])
        
        self.indx2awns = torch.load(self.model_path+'indx2class.pth')# id -> awns(str)
        for idx in self.indx2awns: self.indx2awns[idx] = self.indx2awns[idx][:-1]
        for idx in self.indx2awns: 
            if self.indx2awns[idx]=='AWNLESS ': self.indx2awns[idx] = 'AWNLESS'
        print(self.indx2awns)
        self.awns2indx = {v:k for k,v in self.indx2awns.items()}
            
    ##################################################    
    # step 2: predcsvs(switch mode here)
    ##################################################  
    def predcsvs(self):
        for file in self.csvs:
            print('-'*80+'\nProcessing file: {}'.format(file))
            self.df = pd.read_csv(file)
            self.readcsv(file)
            # self.predImg(file)############ comment this line when check result only
            self.statcsv(file)
            
    def readcsv(self, file):
        # extract info
        df_Paths = pd.Series.to_dict(self.df['image_file_name'])  # dict type
        df_Plots = pd.Series.to_dict(self.df['plot_id'])          # dict type
        df_AWNS  = pd.Series.to_dict(self.df['AWNS'])             # dict type
        df_Cpose = pd.Series.to_dict(self.df['camera_sn'])        # dict type Camera #5: 0671720638(nadir)
        
        self.df_date  = df_Paths[1].split('_')[1]                 # extract the date of the images file
        self.prediction = {i:'None' for i in df_Paths}
        
        print(len(df_Paths))
        # transfer the dict
        for k in df_Plots: df_Plots[k] = df_Plots[k][6:]
        
        # screen nadir images
        self.dinfo = {idx:self.src+path for idx, path in df_Paths.items() if df_Cpose[idx] != 'CAM_0671720638'}      
        
        # plot to image indexs
        self.plot2idx = invDict(df_Plots)# plots_id -> image NO.
    
    def predImg(self,file):
        b_size = 150
        w_size = 25

        dsets = ImageReader(self.dinfo, self.transforms)
        S_sampler = SequentialSampler(dsets)
        dataLoader = torch.utils.data.DataLoader(dsets, batch_size=b_size, sampler=S_sampler, num_workers=w_size, drop_last = False)
        print(len(dsets))

        #predict images in batches
        for data in dataLoader:
            # get the inputs
            img_bt, idx_bt = data
            
            output = self.model(Variable(img_bt.cuda(),volatile = True))
            _, pre_bt = torch.max(output.data, 1)
            pre_bt = pre_bt.cpu().view(-1)
            
            for idx, pre in zip(idx_bt,pre_bt): 
                print('{}:{}'.format(idx,self.indx2awns[pre]),end='\r')
                self.prediction[idx] = self.indx2awns[pre]

        self.df['predawns'] = pd.Series(self.prediction)
        self.df.to_csv(self.dst+'csvs/'+os.path.basename(file)[:-4]+'PredAWN.csv' )
        
        
    def statcsv(self,file):
        # loading prediction
        self.df = pd.read_csv(self.dst + 'csvs/' + os.path.basename(file)[:-4] + 'PredAWN.csv')
        self.df_Preds = pd.Series.to_dict(self.df['predawns'])  # dict type
 
        acc_ct = {p:[] for p in self.plot2idx}# plot_id -> awns' acc info

        for plot, idxlist in self.plot2idx.items():
            if plot not in self.plot2awns: continue
            # plot label
            lab = self.plot2awns[plot]# str
            # check each image in the plot
            for idx in idxlist:
                # image prediction  
                pre = self.df_Preds[idx]# str
                if pre=='None': continue 
                
                acc_ct[plot].append((lab==pre)*1)
                if lab not in AWNS2: continue
                self.cfmat[self.awns2indx[lab],self.awns2indx[pre]] += 1
                if lab==pre:
                    self.acc[0]+=1
                self.acc[1]+=1
            
        # calculate the acc for each date
        for plot, acc_list in acc_ct.items(): 
            if len(acc_list)==0:continue
            self.plot2date2acc[plot][self.df_date] = sum(acc_list)/len(acc_list)
            self.plot2date2num[plot][self.df_date] = len(acc_list)
            
            
    def report(self):
        print('numbers of plots: {}'.format(len(self.plot2awns)))
        writer = pd.ExcelWriter(self.dst + 'plot2date2acc&num.xlsx')
        df_acc = pd.DataFrame(self.plot2date2acc).T
        df_acc['AWNS'] = pd.Series(self.plot2awns)
        df_acc.to_excel(writer,'Sheet1')
        df_num = pd.DataFrame(self.plot2date2num).T
        df_num['AWNS'] = pd.Series(self.plot2awns)
        df_num.to_excel(writer,'Sheet2')
        writer.save()

    def run(self):
        print('-'*80+'\nSetting model')
        self.setModel()
        print('-'*80+'\nPredicting the result')
        self.predcsvs()
        print('-'*80+'\nReporting the result')
        self.report()
        print(self.acc[0]/self.acc[1])
        matrixPlot(self.cfmat, self.dst, 'test')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        