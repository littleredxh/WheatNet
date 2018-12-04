import KNet_DataPrepareAwn as dataPrep
import KNet_keyfile as keyfile
import KNet_Train as K

import shutil, os, torch
from multiprocessing import Pool
from glob import glob
import time

PHASE = ['train','val']
AWNS = ['AWNED','AWNLESS ']
        
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
    dst = 'awns2/'
    if i!=8:
        shutil.rmtree('awns2/')
        # perpare the training images
        csvfiles = keyfile.csvfiles_awns

        data = dataPrep.csv2img(src,dst,csvfiles)
        cpList = data.run()

        if __name__ == '__main__':
            for phase in PHASE:
                for x in AWNS: 
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
    dst = 'awns2_result/'+str(i)+'/'
    gpuid = [0,1]
    L = K.learn(src, dst, gpuid)
    record = L.run()
    record_all.append(record)
    
torch.save(record_all,dst+'record_all.pth')


import KNet_keyfile
csvfiles = KNet_keyfile.csvfiles_pctA

import KNet_TimeSeriesAwn
src = '/project/focus/hong/AllData/'
dst = 'awns2_result/'

model_path = 'awns2_result/0/model.pth'
# perpare the training images
ans = KNet_TimeSeriesAwn.csv2pred(src,dst,csvfiles,model_path)
ans.run()

