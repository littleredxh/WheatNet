import numpy as np
import pandas as pd
import os, copy, random, torch
from KNet_utils import invDict,combDict1D

PHASE = ['train','val']
AWNS = ['AWNED','AWNLESS ']
    
class csv2img:
    def __init__(self,src,dst,csvfiles):
        self.src = src
        self.dst = dst
        self.csvs = csvfiles# a list
        if not os.path.exists(self.dst): os.makedirs(self.dst)
        
        self.plot2awns = {}# plot_id -> awns(str)
        self.plot2dir = {}# plot_id -> images dir
        
    def run(self):
        self.readfiles()
        self.spPlots()
        self.cpFile()
        self.report()
        torch.save(self.plot2awns,'awns2_result/plot2awns.pth')
        return self.cpList

    def readfiles(self):
        for file in self.csvs:
            print('Loading file: {}'.format(file))
            self.readfile(file)
            
    def readfile(self, file):
        df = pd.read_csv(file)
        fileNames = pd.Series.to_dict(df['image_file_name'])  # dict type
        file_AWNS = pd.Series.to_dict(df['AWNS'])             # dict type
        filePlots = pd.Series.to_dict(df['plot_id'])          # dict type
        fileCpose = pd.Series.to_dict(df['camera_sn'])        # dict type Camera #5: 0671720638(nadir)
        
        self.plot = sorted(set([v for k,v in filePlots.items()]))# unique plots_id
        plot2idx = invDict(filePlots)
        plot2dir = {p:[] for p in self.plot}#plot_id -> images dir
        
        for i, a in file_AWNS.items():
            if filePlots[i] in self.plot2awns:
                if self.plot2awns[filePlots[i]] != a: print('wrong info')
            else:
                self.plot2awns[filePlots[i]] = a
            
        for plot, idxlist in plot2idx.items():
            for idx in idxlist:
                if fileCpose[idx] == 'CAM_0671720638': continue
                plot2dir[plot].append(fileNames[idx]) 

        self.plot2dir = combDict1D(self.plot2dir, plot2dir)

    def report(self):
        print('-'*100)
        print('numbers of plots: {}'.format(len(self.plot2awns)))
        for awns in AWNS:
            print(len(self.awn2plots[awns]))
        
        print('-'*100)
        maxnum = 7000
        minnum = 700
        for phase in PHASE:
            for awns in AWNS:
                if len(self.cpList[phase][awns])>maxnum and phase==PHASE[0]:
                    self.cpList[phase][awns] = random.sample(self.cpList[phase][awns],maxnum)
                if len(self.cpList[phase][awns])>minnum and phase==PHASE[1]:
                    self.cpList[phase][awns] = random.sample(self.cpList[phase][awns],minnum)
                
                print('{}-{}: {}'.format(phase,awns,len(self.cpList[phase][awns])))
                
                
    def spPlots(self):
        self.awn2plots = invDict(self.plot2awns)
        self.phase2awns2plots={p:{a:[] for a in AWNS} for p in PHASE}
        for awns, plots in self.awn2plots.items():
            random.shuffle(plots)
            if awns==AWNS[0]:# AWNED
                self.phase2awns2plots[PHASE[0]][awns] = plots[:-70]
                self.phase2awns2plots[PHASE[1]][awns] = plots[-70:]
            if awns==AWNS[1]:# AWNLESS
                self.phase2awns2plots[PHASE[0]][awns] = plots[:-5]
                self.phase2awns2plots[PHASE[1]][awns] = plots[-5:]

            
    def cpFile(self):
        self.cpList = {p:{a:[] for a in AWNS} for p in PHASE}
        a = 100# limited size
        for phase in PHASE:
            for awns in AWNS:
                if awns==AWNS[0]:a =  6# limited size
                if awns==AWNS[1]:a = 100# limited size
                for plot in self.phase2awns2plots[phase][awns]:
                    if len(self.plot2dir[plot])>a:
                        dir_sample = [self.plot2dir[plot][i] for i in sorted(random.sample(range(len(self.plot2dir[plot])), a))]
                    else:
                        dir_sample = self.plot2dir[plot]
                        
                    for img in dir_sample:
                        if os.path.isfile(self.src+img):
                            dst = self.dst+phase+'/'+awns+'/'
                            if not os.path.exists(dst): os.makedirs(dst)
                            self.cpList[phase][awns].append((self.src+img, dst+os.path.basename(img)))
                        else:
                            print('no file',end='\r')
 
                
# import KNet_keyfile
# csvfiles = KNet_keyfile.csvfiles_awns
# src = '/project/focus/hong/AllData/'
# dst = 'awns/'
# # perpare the training images
# ans = csv2img(src,dst,csvfiles)
# ans.run()