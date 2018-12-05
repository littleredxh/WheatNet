import pandas as pd
import os, random

PHASE = ['tra','val']
PCT = [0,10,20,30,40,50,60,70,80,90,100]
    
class csv2img:
    def __init__(self, src, dst, csvfiles):
        if not os.path.exists(dst): os.makedirs(dst)
        self.src = src
        self.dst = dst
        self.csvs = csvfiles# list
        self.plot = None
        self.data_dict = {p:{x:[] for x in PCT} for p in PHASE}
        
    def run(self):
        self.loadfile()
        self.spPlots()
        self.report()
        return self.data_dict

    def loadfile(self):
        # extracting plots_id
        plot_id = []
        for file in self.csvs['17']:
            filePlots = pd.Series.to_dict(pd.read_csv(file)['plot_id'])          # dict type
            plot_id += [v for k,v in filePlots.items()]
            
        self.plot = sorted(set(plot_id))# unique plots_id
            
        # building data dict
        plot2pct2dir = {p:{pct:[] for pct in PCT} for p in self.plot}
        for file in self.csvs['17']:
            print('Loading file: {}'.format(file))
            df = pd.read_csv(file)
            fileNames = pd.Series.to_dict(df['image_file_name'])  # dict type
            filePlots = pd.Series.to_dict(df['plot_id'])          # dict type
            filePCThd = pd.Series.to_dict(df['PCTHEAD'])          # dict type
            fileCpose = pd.Series.to_dict(df['camera_sn'])        # dict type Camera #5: 0671720638(nadir)
            # transfer the dict
            # for k in filePlots: filePlots[k] = filePlots[k][6:]

            # building the dict
            for idx, plot in filePlots.items():
                if fileCpose[idx] == 'CAM_0671720638': continue
                for pct in PCT: 
                    if pct-4 < filePCThd[idx] and filePCThd[idx] < pct+4: 
                        plot2pct2dir[plot][pct].append(self.src+fileNames[idx]) 
                        
        self.plot2pct2dir = plot2pct2dir

    def spPlots(self):
        random.shuffle(self.plot)
        # print(self.plot)
        plot_sep = {'tra':self.plot[:-100], 'val':self.plot[-100:]}
        for phase in PHASE:
            for plot in plot_sep[phase]:
                for pct in PCT:
                    for img in self.plot2pct2dir[plot][pct]:
                        if os.path.isfile(img):
                            self.data_dict[phase][pct].append(img)
                        else:
                            print('no file')

    def report(self):
        print('-'*100)
        print('numbers of plots: {}'.format(len(self.plot)))

        print('-'*100)
        maxnum = 2000
        minnum = 200
        for phase in PHASE:
            for pct in PCT:
                print('{}-{:03}: {}'.format(phase,pct,len(self.data_dict[phase][pct])))
            