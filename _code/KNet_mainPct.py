from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
from sklearn.preprocessing import normalize
import glob, os, time, copy

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms, models

# import .KNet_utils as K
from .KNet import KNet

PHASE = ['train', 'val']
RGBmean, RGBstdv = [0.429, 0.495, 0.259], [0.218, 0.224, 0.171]  

def norml1(vec):# input N by F
    F = vec.size(1)
    w = torch.t(vec.sum(1).repeat(F,1))
    return vec.div(w)

class learn():
    def __init__(self, src, dst, gpuid):
        self.src = src
        self.dst = dst
        self.gpuid = gpuid
        
        if len(gpuid)>1: 
            self.mp = True
        else:
            self.mp = False
            
        self.batch_size = 22
        self.num_workers = 12
        
        self.init_lr = 0.001
        self.decay_epoch = 10
        
        self.num_features = 11
        self.criterion = nn.CrossEntropyLoss()
        self.record = []
        
        
    def run(self):
        if not self.setsys(): return
        self.loadData()
        self.setModel()
        self.printInfo()
        self.num_epochs=20
        self.decay_time =[False, False]
        self.opt(self.num_epochs)
    
    
    def setsys(self):
        if not os.path.exists(self.src): print('src folder not exited'); return False
        if not torch.cuda.is_available(): print('No GPU detected'); return False
        if not os.path.exists(self.dst): os.makedirs(self.dst)
        torch.cuda.set_device(self.gpuid[0]); print('Current device is GPU: {}'.format(torch.cuda.current_device()))
        return True
    
    
    def loadData(self):
        data_transforms = {'train': transforms.Compose([
                                    transforms.Resize(int(224*3*1.1)),
                                    transforms.RandomCrop(224*3),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(RGBmean, RGBstdv)]),
                             'val': transforms.Compose([
                                    transforms.Resize(int(224*3*1.1)),
                                    transforms.CenterCrop(224*3),
                                    transforms.ToTensor(),
                                    transforms.Normalize(RGBmean, RGBstdv)])}
        
        self.dsets = {p: datasets.ImageFolder(os.path.join(self.src, p), data_transforms[p]) for p in PHASE}
        self.class2indx = self.dsets['train'].class_to_idx
        self.indx2class = {v: k for k,v in self.class2indx.items()}
        self.class_size = {p: {k: 0 for k in self.class2indx} for p in PHASE }# number of images in each class
        self.N_classes = len(self.class2indx)# total number of classes
        self.bookmark = {p:[] for p in PHASE}# index bookmark
        
        # number of images in each class
        for phase in PHASE:
            for key in self.class2indx:
                filelist = [f for f in glob.glob(self.src + phase + '/' + key + '/' + '*.JPG')]
                self.class_size[phase][key] = len(filelist)

        # index bookmark
        for phase in PHASE:
            sta,end = 0,0
            for idx in sorted(self.indx2class):
                classkey = self.indx2class[idx]
                end += self.class_size[phase][classkey]
                self.bookmark[phase].append((sta,end))
                sta += self.class_size[phase][classkey]
        return

    def setModel(self):
        # create whole model
        Kmodel = KNet(self.N_classes)
        
        # parallel computing and opt setting
        if self.mp:
            print('Training on Multi-GPU')
            self.batch_size = self.batch_size*len(self.gpuid)
            self.model = torch.nn.DataParallel(Kmodel,device_ids=self.gpuid).cuda()#
#             self.model = torch.load('pct6_result/result13_paper/model.pth')
#             print(self.model)
            self.optimizer = optim.SGD(self.model.module.parameters(), lr=0.01, momentum=0.9)
        else: 
            print('Training on Single-GPU')
            self.model = Kmodel.cuda()
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        return

    
    def printInfo(self):
        print('\nimages size of each class\n'+'-'*50)
        print(self.class_size)
        print('\nclass to index\n'+'-'*50)
        print(self.dsets['train'].class_to_idx)
        print('\nbookmark\n'+'-'*50)
        print(self.bookmark)
        return
    
    
    def lr_scheduler(self, epoch):
        if epoch>=0.5*self.num_epochs and not self.decay_time[0]: 
            self.decay_time[0] = True
            lr = self.init_lr*0.1
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        if epoch>=0.8*self.num_epochs and not self.decay_time[1]: 
            self.decay_time[1] = True
            lr = self.init_lr*0.01
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        return

    
    ##################################################
    # step 3: Learning
    ##################################################
    def tra(self):
        if self.mp:
            self.model.module.train(True)  # Set model to training mode
        else:
            self.model.train(True)  # Set model to training mode
             
        dataLoader = torch.utils.data.DataLoader(self.dsets['train'], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)#sampler=XSampler(self.classMark['tra'], GSize=1)
        
        L_data, T_data, N_data = 0.0, 0, 0
        accMat = torch.zeros(self.N_classes,self.N_classes)
        # iterate batch
        for data in dataLoader:
            self.optimizer.zero_grad()
            inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
            fvec = self.model(Variable(inputs_bt.cuda()))
            
            loss = self.criterion(fvec, Variable(labels_bt).cuda())
            
            loss.backward()
            self.optimizer.step()  

            _, preds_bt = torch.max(fvec.cpu().data, 1)
            
            L_data += loss.data[0]
            T_data += torch.sum(preds_bt.view(-1) == labels_bt)
            N_data += len(labels_bt)
            for i in range(len(labels_bt)): accMat[labels_bt[i],preds_bt[i]] += 1
                
        return L_data/N_data, T_data/N_data, norml1(accMat)

    def val(self):
        if self.mp:
            self.model.module.train(False)  # Set model to training mode
        else:
            self.model.train(False)  # Set model to training mode
            
        dataLoader = torch.utils.data.DataLoader(self.dsets['val'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        L_data, T_data, N_data = 0.0, 0, 0
        accMat = torch.zeros(self.N_classes,self.N_classes)
        # iterate batch
        for data in dataLoader:
            inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
            
            fvec = self.model(Variable(inputs_bt.cuda()))
            
            loss = self.criterion(fvec, Variable(labels_bt).cuda())
            
            _, preds_bt = torch.max(fvec.cpu().data, 1)

            L_data += loss.data[0]
            T_data += torch.sum(preds_bt.view(-1) == labels_bt)
            N_data += len(labels_bt)
            for i in range(len(labels_bt)): accMat[labels_bt[i],preds_bt[i]] += 1
                
        return L_data/N_data, T_data/N_data, norml1(accMat)
        
    def opt(self, num_epochs):
        # recording time and epoch acc and best result
        since = time.time()
        self.best_epoch = 0
        self.best_acc = 0
        for epoch in range(num_epochs):
            print('Epoch {}/{} \n '.format(epoch, num_epochs - 1) + '-' * 40)
            self.lr_scheduler(epoch)
            
            tra_loss, tra_acc, tra_conf = self.tra()
            val_loss, val_acc, val_conf = self.val()
            
            self.record.append((epoch, tra_loss, val_loss, tra_acc, val_acc))
            print('tra - Loss:{:.4f} - Acc:{:.4f}\nval - Loss:{:.4f} - Acc:{:.4f}'.format(tra_loss, tra_acc, val_loss, val_acc))    
    
            torch.save(tra_conf, self.dst + 'traConf_{:02}.pth'.format(epoch))
            torch.save(val_conf, self.dst + 'valConf_{:02}.pth'.format(epoch))
            # deep copy the model
            if epoch >= 1 and val_acc> self.best_acc:
                self.best_acc = val_acc
                self.best_epoch = epoch
                torch.save(self.model, self.dst + 'model.pth')
        
        torch.save(self.record, self.dst + 'record.pth')
        
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
        print('Best val acc in epoch: {}'.format(self.best_epoch))
        return
   

    