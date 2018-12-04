import os
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

from KNet_utils import invDict, matrixPlot,combDict3D
from KNet_Loader import ImageReader

PHASE = ['train','val']
PCT = [0,10,20,30,40,50,60,70,80,90,100]
# normalization paras
RGBmean, RGBstdv = [0.429, 0.495, 0.259], [0.218, 0.224, 0.171]  

def combDict_plot2date2pct(dict_main,dict_in):
    # plot_id -> date -> pct
    if dict_main == None: dict_main = dict_in
    else:
        for plot in dict_in:
            if plot in dict_main: 
                for date in dict_in[plot]: 
                    if not date in dict_main[plot]: 
                        dict_main[plot].update({date : dict_in[plot][date]}) 
            else: dict_main[plot] = dict_in[plot]
    return dict_main


def checkfile(path):
    try:
        img = default_loader(path)
    except OSError:
        if os.path.exists(path):
            os.remove(path)
        print('OSError')
    return
    
class csv2pred:
    ##################################################
    # initialization
    ##################################################
    def __init__(self,src,dst,csvfiles,model_path):
        self.src = src
        self.dst = dst
        self.csvs = csvfiles# a list a csv files
        self.plot2pct2path = None# plot_id -> pct -> images dir
        self.plot2date2pct_lab = None# plot_id -> date -> pct
        self.plot2date2pct_pre = None# plot_id -> date -> pct
        self.model_path = model_path
        if not os.path.exists(self.dst): os.makedirs(self.dst)
        if not os.path.exists(self.dst + 'csvs/'): os.makedirs(self.dst + 'csvs/')
        self.acc = [0,0]
        self.cfmat = np.zeros(( len(PCT), len(PCT) ))
        
    ##################################################
    # util
    ##################################################
    def combDict_plot2pct2path(self,dict_in):
        # plot_id -> pct -> images dir
        if self.plot2pct2path == None: self.plot2pct2path = dict_in
        else:
            for plot in dict_in:
                if plot in self.plot2pct2path: 
                    for pct in PCT: self.plot2pct2path[plot][pct] += dict_in[plot][pct] 
                else: self.plot2pct2path[plot] = dict_in[plot]
    

    
    ##################################################
    # step 1: setModel
    ##################################################
    def setModel(self):
        # load model
        self.model = torch.load(self.model_path).module
        self.transforms = transforms.Compose([transforms.Scale(224*4),
                                              transforms.CenterCrop(224*3),
                                              transforms.ToTensor(),
                                              transforms.Normalize(RGBmean, RGBstdv)])
    
    ##################################################    
    # report
    ##################################################
    def report(self):
        print('-'*100)
        print('numbers of plots: {}'.format(len(self.plot2date2pct_lab)))
        df_pct_lab = pd.DataFrame(self.plot2date2pct_lab).T
        df_pct_pre = pd.DataFrame(self.plot2date2pct_pre).T
        writer = pd.ExcelWriter(self.dst + 'pct_lab.xlsx')
        df_pct_lab.to_excel(writer,'Sheet1')
        writer.save()
        
        writer = pd.ExcelWriter(self.dst + 'pct_pre.xlsx')
        df_pct_pre.to_excel(writer,'Sheet1')
        writer.save()
    
    ##################################################    
    # step 2: predcsvs(switch mode here)
    ##################################################
    def predcsvs(self):
        for file in self.csvs:
            print('Processing file: {}'.format(file))
            self.df = None
            self.readcsv(file)
            # self.precheck()## only need for the first time
            # self.predImg(file)############ comment this line when check result only
            self.statcsv(file)
            
    def readcsv(self, file):
        # use pd lib to read csv files
        self.df = pd.read_csv(file)
        # extract info
        self.df_Paths = pd.Series.to_dict(self.df['image_file_name'])  # dict type
        self.df_PCThd = pd.Series.to_dict(self.df['PCTHEAD'])          # dict type
        self.df_Plots = pd.Series.to_dict(self.df['plot_id'])          # dict type
        self.df_Cpose = pd.Series.to_dict(self.df['camera_sn'])        # dict type Camera #5: 0671720638(nadir)

    def precheck(self):
        dinfo = {idx:self.src+path for idx, path in self.df_Paths.items()}
        # check broken images
        path = [path for _, path in dinfo.items()]
        Pool(32).map(checkfile,path)
        
    def predImg(self,file):#
        b_size=150
        prediction = np.ones(len(self.df['image_file_name']))*(-1)
        
        # screen nadir images
        dinfo = {idx:self.src+path for idx, path in self.df_Paths.items() if self.df_Cpose[idx] != 'CAM_0671720638'}      
        dsets = ImageReader(dinfo, self.transforms)
        S_sampler = SequentialSampler(dsets)
        dataLoader = torch.utils.data.DataLoader(dsets, batch_size=b_size, sampler=S_sampler, num_workers=25, drop_last = False)
        print(len(dsets))
        
        #predict images in batches
        for data in dataLoader:
            # get the inputs
            img_bt, idx_bt = data
            
            output = self.model(Variable(img_bt.cuda(),volatile = True))
            _, pre_bt = torch.max(output.data, 1)
            pre_bt = pre_bt.cpu().view(-1)*10
            
            for idx, pre in zip(idx_bt,pre_bt): 
                print('{}:{}'.format(idx,pre),end='\r')
                if idx==pre:
                    self.acc[0] += 1
                self.acc[1] += 1
                prediction[idx] = pre

        self.df['pred'] = pd.Series(prediction, index=self.df.index)
        self.df.to_csv(self.dst+'csvs/'+os.path.basename(file)[:-4]+'Pred.csv' )
        
        
    def statcsv(self,file):
        self.df = pd.read_csv(self.dst+'csvs/'+os.path.basename(file)[:-4]+'Pred.csv')
        self.df_preds = pd.Series.to_dict(self.df['pred'])
        # unique plots_id
        self.plot = sorted(set([v for k,v in self.df_Plots.items()]))
        # plots_id -> image NO.
        plot2idx = invDict(self.df_Plots)
        #plot_id -> pct -> images dir
        plot2pct2path = {p:{} for p in self.plot}
        #plot_id -> date -> pct
        plot2date2pct_lab = {p:{} for p in self.plot}
        plot2date2pct_pre = {p:{} for p in self.plot}
        
        for plot, idxlist in plot2idx.items():
            pct2path = {x:[] for x in PCT}
            date2pct_lab = {}
            date2pct_pre = {}
            
            for idx in idxlist:
                # screen the nadir images
                if self.df_Cpose[idx] == 'CAM_0671720638': continue
                
                # add items into pct2path for a given plot
                for x in PCT:
                    if x-3 < self.df_PCThd[idx] and self.df_PCThd[idx] < x+3: 
                        pct2path[x].append(self.df_Paths[idx])
                        self.cfmat[int(x/10),int(self.df_preds[idx]/10)] += 1
                
                        if x==self.df_preds[idx]:
                            self.acc[0] += 1
                self.acc[1] += 1
                
                # extract the date of the image
                date = self.df_Paths[idx].split('_')[1]
                # add items into date2pct for a given plot
                if not date in date2pct_lab:
                    date2pct_lab.update({date : [self.df_PCThd[idx]]})
                    date2pct_pre.update({date : [self.df_preds[idx]]})
                else: 
                    date2pct_lab[date] += [self.df_PCThd[idx]]
                    date2pct_pre[date] += [self.df_preds[idx]]
            
            # calculate the mean of pct for each date
            for date in date2pct_lab: 
                lablist = [i for i in date2pct_lab[date] if i>=0]
                prelist = [i for i in date2pct_pre[date] if i>=0]
                
                if len(lablist)<1:
                    date2pct_lab[date] = None
                else:
                    date2pct_lab[date] = mean(lablist)
                
                if len(prelist)<1:
                    date2pct_pre[date] = None
                else:
                    date2pct_pre[date] = mean(prelist)
                
            # add item to dict    
            plot2pct2path[plot] = pct2path
            plot2date2pct_lab[plot] = date2pct_lab
            plot2date2pct_pre[plot] = date2pct_pre
        
        # combine dict
        self.combDict_plot2pct2path(plot2pct2path)
        self.plot2date2pct_lab = combDict_plot2date2pct(self.plot2date2pct_lab, plot2date2pct_lab)
        self.plot2date2pct_pre = combDict_plot2date2pct(self.plot2date2pct_pre, plot2date2pct_pre)
    
    
            
    ##################################################    
    # timeSeriesPlot
    ##################################################                         
    def timeSeriesPlot(self):
        if not os.path.exists(self.dst + 'timeS/'): os.makedirs(self.dst + 'timeS/')
        for plot in self.plot2date2pct_lab:
            date = []
            pctP = []
            pctL = []
            for key in self.plot2date2pct_lab[plot]: 
                date.append(key)
                pctL.append(self.plot2date2pct_lab[plot][key])
                pctP.append(self.plot2date2pct_pre[plot][key])

            x = [dt.datetime.strptime(d,'%Y%m%d').date() for d in date]
            yP = pctP
            yL = pctL

            fig, ax = plt.subplots(1)
            fig.autofmt_xdate()
            plt.plot(x,yP,'ro')
            plt.plot(x,yL,'b*')
            plt.ylim(0, 100)
            plt.legend(['Pred', 'Label'], loc='upper left')
            
            xfmt = mdates.DateFormatter('%m-%d')
            ax.xaxis.set_major_formatter(xfmt)
            plt.savefig(self.dst+'timeS/'+plot)
            plt.close()
            
    ##################################################    
    # all
    ##################################################  
    def run(self):
        self.setModel()
        self.predcsvs()
        self.report()
        # self.timeSeriesPlot()
        print(self.acc[0]/self.acc[1])
        matrixPlot(self.cfmat, self.dst, 'test')

        
