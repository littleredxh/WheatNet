import numpy as np
import pandas as pd
import os, copy, random
from KNet_utils import invDict

PHASE = ['train','val']
PCT = [0,10,20,30,40,50,60,70,80,90,100]
    
class csv2img:
    def __init__(self, dst, csvfiles):
        if not os.path.exists(dst): os.makedirs(dst)
        self.src = '/project/focus/hong/AllData/'
        self.dst = dst
        self.csvs = csvfiles# list
        
    def combDict(self,dict_in):
        # plot_id -> pct -> images dir
        if self.dict == None: self.dict = dict_in
        else:
            for plot in dict_in:
                if plot in self.dict: 
                    for pct in PCT:
                        self.dict[plot][pct] += dict_in[plot][pct] 
                else: self.dict[plot] = dict_in[plot]

    def run(self):
        self.dict = None# plot_id -> pct -> images dir
        self.readfiles()# get self,dict
        self.spPlots()
        self.cpFile()
        self.report()
        return self.cpList# [0:[list m000],10:[list m010]...]

    def readfiles(self):
        for file in self.csvs['17']:
            print('Loading file: {}'.format(file))
            self.readfile(file)
        # for file in self.csvs['16']:
        #     print('Loading file: {}'.format(file))
        #     self.readfile16(file)
            
    def readfile(self, file):
        df = pd.read_csv(file)
        fileNames = pd.Series.to_dict(df['image_file_name'])  # dict type
        filePlots = pd.Series.to_dict(df['plot_id'])          # dict type
        filePCThd = pd.Series.to_dict(df['PCTHEAD'])          # dict type
        fileCpose = pd.Series.to_dict(df['camera_sn'])        # dict type Camera #5: 0671720638(nadir)
        
        # transfer the dict
        # for k in filePlots: filePlots[k] = filePlots[k][6:]
            
        # plots_id
        self.plot = sorted(set([v for k,v in filePlots.items()]))# unique plots_id
        plot2idx = invDict(filePlots)
        plot2pct2dir = {p:{} for p in self.plot}

        for plot, idxlist in plot2idx.items():
            pct2dir = {p:[] for p in PCT}
            for idx in idxlist:
                if fileCpose[idx] == 'CAM_0671720638': continue
                for pct in PCT: 
                    if pct-4 < filePCThd[idx] and filePCThd[idx] < pct+4: pct2dir[pct].append(fileNames[idx]) 
            
            plot2pct2dir[plot] = pct2dir

        self.combDict(plot2pct2dir)
        
    def readfile16(self, file):
        df = pd.read_csv(file)
        fileNames = pd.Series.to_dict(df['image_file_name'])  # dict type
        filePlots = pd.Series.to_dict(df['plot_id'])          # dict type
        filePCThd = pd.Series.to_dict(df['PCTHEAD'])          # dict type
        fileCpose = pd.Series.to_dict(df['camera_sn'])        # dict type Camera #5: 0671720638(nadir)
        
        # transfer the dict
        for k in filePlots: filePlots[k] = filePlots[k][6:]
            
        # plots_id
        self.plot = sorted(set([v for k,v in filePlots.items()]))# unique plots_id
        plot2idx = invDict(filePlots)
        plot2pct2dir = {p:{} for p in self.plot}

        for plot, idxlist in plot2idx.items():
            pct2dir = {p:[] for p in PCT}
            for idx in idxlist:
                if fileCpose[idx] == 'CAM_0671720638': continue
                for pct in PCT:
                    if pct == 70 or pct ==80 or pct ==90: continue
                    # if pct == filePCThd[idx]: pct2dir[pct].append(fileNames[idx]) 
                    if pct-4 < filePCThd[idx] and filePCThd[idx] < pct+4: pct2dir[pct].append(fileNames[idx]) 
            
            plot2pct2dir[plot] = pct2dir

        self.combDict(plot2pct2dir)
        
    def report(self):
        print('-'*100)
        print('numbers of plots: {}'.format(len(self.dict_sp)))

        print('-'*100)
        maxnum = 2000
        minnum = 200
        for phase in PHASE:
            for pct in PCT:
                if len(self.cpList[phase][pct])>maxnum and phase==PHASE[0]:
                    self.cpList[phase][pct] = random.sample(self.cpList[phase][pct],maxnum)
                if len(self.cpList[phase][pct])>minnum and phase==PHASE[1]:
                    self.cpList[phase][pct] = random.sample(self.cpList[phase][pct],minnum)
                
                print('{}-{}: {}'.format(phase,pct,len(self.cpList[phase][pct])))
                
    
    def spPlots(self):
        self.dict_sp = copy.deepcopy(self.dict)
        for plot in self.dict:
            for pct in PCT:
                a=100
                if pct==0:
                    # a=10
                    if len(self.dict[plot][pct])>a:
                        dir_sample = [self.dict[plot][pct][i] for i in sorted(random.sample(range(len(self.dict[plot][pct])), a))]
                    else:
                        dir_sample = self.dict[plot][pct]
                elif pct==100:
                    # a=8
                    if len(self.dict[plot][pct])>a:
                        dir_sample = [self.dict[plot][pct][i] for i in sorted(random.sample(range(len(self.dict[plot][pct])), a))]
                    else:
                        dir_sample = self.dict[plot][pct]
                else:
                    # a=20
                    if len(self.dict[plot][pct])>a:
                        dir_sample = [self.dict[plot][pct][i] for i in sorted(random.sample(range(len(self.dict[plot][pct])), a))]
                    else:
                        dir_sample = self.dict[plot][pct]
                    
                self.dict_sp[plot][pct] = dir_sample

    def cpFile(self):
        random.shuffle(self.plot)
        # print(self.plot)
        learnSet = {'train':self.plot[:-100], 'val':self.plot[-100:]}
        self.cpList = {p:{x:[] for x in PCT} for p in PHASE}

        for phase in PHASE:
            for plot in learnSet[phase]:
                for pct in PCT:
                    # if self.dict_sp[plot][pct] != None: ##
                    for img in self.dict_sp[plot][pct]:
                        if os.path.isfile(self.src+img):
                            if pct==0: 
                                dst = self.dst+phase+'/'+'m00'+str(pct)+'/'
                            elif pct==100: 
                                dst = self.dst+phase+'/'+'m'+str(pct)+'/'
                            else:  
                                dst = self.dst+phase+'/'+'m0'+str(pct)+'/'

                            if not os.path.exists(dst): os.makedirs(dst)
                            self.cpList[phase][pct].append((self.src+img, dst+os.path.basename(img)))
                        else:
                            print('no file',end='\r')
        
         
            
            
            
            
            
            
            