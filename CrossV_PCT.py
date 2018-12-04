import KNet_DataPreparePct as dataPrep
import KNet_keyfile as keyfile
import _code.KNet_mainPct as K

import shutil, os, torch
from multiprocessing import Pool
from glob import glob
import time

PHASE = ['train','val']
PCT = [0,10,20,30,40,50,60,70,80,90,100]
        
def cpImg(t):
    if os.path.isfile(t[1]):
        return
    else:
        shutil.copy2(t[0], t[1])
    # print('copied: '+ t[1])
    
N = 1
record_all =[]
for i in range(N):
    src = '/project/focus/hong/AllData/'
    dst = 'pct6/'
    if i!=0:
        shutil.rmtree('pct6/')
        # perpare the training images
        csvfiles = {'17':keyfile.csvfiles_pctA,'16':keyfile.csvfiles_pctA16}

        data = dataPrep.csv2img(src,dst,csvfiles)
        cpList = data.run()

        for phase in PHASE:
            for x in PCT: 
                print(x,end='\r')
                pool = Pool(processes=10)
                pool.map(cpImg, cpList[phase][x])
                pool.terminate()
        print('data prepared!')

        for subfolder in sorted(glob(dst+'train/*/')):
            print(subfolder)
            os.rename(subfolder, subfolder[:-1]+'0/')
            time.sleep(0.1)
        for subfolder in sorted(glob(dst+'val/*/')):
            os.rename(subfolder, subfolder[:-1]+'0/')
            time.sleep(0.1)
    # train
    src = dst
    dst = 'pct6_f_result/'+str(i)+'/'
    gpuid = [0,1]
    L = K.learn(src, dst, gpuid)
    record = L.run()
#     record_all.append(record)
    
# torch.save(record_all,dst+'record_all.pth')