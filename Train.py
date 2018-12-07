import os, time, copy, random
from glob import glob

from torchvision import models, transforms, datasets
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch

import Utils as K
from Net import KNet
from Reader import  ImageReader
from Sampler import BalanceSampler
import numpy as np
from sklearn.preprocessing import normalize

PHASE = ['tra', 'val']
RGBmean, RGBstdv = [0.429, 0.495, 0.259], [0.218, 0.224, 0.171]  

class learn():
    def __init__(self, src, dst, data_dict, gpuid=[0,1]):
        self.src = src
        self.dst = dst
        self.gpuid = gpuid
        self.data_dict = data_dict
        
        if len(gpuid)>1: 
            self.mp = True
        else:
            self.mp = False

        self.batch_size = 20
        self.num_workers = 20
        
        self.init_lr = 0.001
        self.decay_time = [False,False]
        self.decay_rate = 0.1
        
        self.num_features = 11
        self.criterion = nn.CrossEntropyLoss()
        self.record = {p:[] for p in PHASE}
        
        
    def run(self,num_epochs):
        if not self.setsys(): return
        self.num_epochs = num_epochs
        self.loadData()
        self.setModel()
        self.train(num_epochs)
                                              
    def setsys(self):
        if not os.path.exists(self.src): print('src folder not exited'); return False
        if not torch.cuda.is_available(): print('No GPU detected'); return False
        if not os.path.exists(self.dst): os.makedirs(self.dst)
        torch.cuda.set_device(self.gpuid[0]); print('Current device is GPU: {}'.format(torch.cuda.current_device()))
        return True   
    
    def loadData(self):
        data_transforms = {'tra': transforms.Compose([
                                  transforms.Resize(224*4),
                                  transforms.RandomCrop(224*3),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(RGBmean, RGBstdv)]),
                           'val': transforms.Compose([
                                  transforms.Resize(224*4),
                                  transforms.CenterCrop(224*3),
                                  transforms.ToTensor(),
                                  transforms.Normalize(RGBmean, RGBstdv)])}
        
        self.dsets = {p:ImageReader(self.data_dict[p], data_transforms[p]) for p in PHASE} 
        self.intervals = self.dsets['tra'].intervals
        self.classSize = len(self.intervals)
        

    def setModel(self):
        # create whole model
        Kmodel = KNet(self.num_features,self.classSize)
        
        # parallel computing and opt setting
        if self.mp:
            print('Training on Multi-GPU')
            self.batch_size = self.batch_size*len(self.gpuid)
            self.model = torch.nn.DataParallel(Kmodel,device_ids=self.gpuid).cuda()#
            self.optimizer = optim.SGD(self.model.module.parameters(), lr=0.01, momentum=0.9)
        else: 
            print('Training on Single-GPU')
            self.model = Kmodel.cuda()
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        return
    
    def lr_scheduler(self, epoch):
        if epoch>=0.6*self.num_epochs and not self.decay_time[0]: 
            self.decay_time[0] = True
            lr = self.init_lr*self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        if epoch>=0.9*self.num_epochs and not self.decay_time[1]: 
            self.decay_time[1] = True
            lr = self.init_lr*self.decay_rate*self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        return

    def train(self, num_epochs):
        # recording time and epoch acc and best result
        since = time.time()
        self.best_acc = 0.0
        self.best_epoch = 0
        for epoch in range(num_epochs):
            print('Epoch {}/{} \n '.format(epoch, num_epochs - 1) + '-' * 40)
            for phase in PHASE:
                # recording the result
                accMat = np.zeros((self.classSize,self.classSize))
                running_loss = 0.0
                N_T, N_A = 0,0
                
                # Adjust the model for different phase
                if phase == 'tra':
                    dataLoader = torch.utils.data.DataLoader(self.dsets[phase], batch_size=self.batch_size, 
                                                     sampler=BalanceSampler(self.intervals, GSize=1), num_workers=self.num_workers)
                    
                    self.lr_scheduler(epoch)
                    if self.mp:
                        self.model.module.train(True)  # Set model to training mode
                        if epoch < int(num_epochs*0.3): self.model.module.R.d_rate(0.2)
                        elif epoch >= int(num_epochs*0.3) and epoch < int(num_epochs*0.6): self.model.module.R.d_rate(0.1)
                        elif epoch >= int(num_epochs*0.6) and epoch < int(num_epochs*0.8): self.model.module.R.d_rate(0.05)
                        elif epoch >= int(num_epochs*0.8): self.model.module.R.d_rate(0)

                    if not self.mp:
                        self.model.train(True)  # Set model to training mode
                        if epoch < int(num_epochs*0.3): self.model.R.d_rate(0.1)
                        elif epoch >= int(num_epochs*0.3) and epoch < int(num_epochs*0.6): self.model.R.d_rate(0.1)
                        elif epoch >= int(num_epochs*0.6) and epoch < int(num_epochs*0.8): self.model.R.d_rate(0.05)
                        elif epoch >= int(num_epochs*0.8): self.model.R.d_rate(0)
                        
                else:
                    dataLoader = torch.utils.data.DataLoader(self.dsets[phase], batch_size=self.batch_size, 
                                                     shuffle=False, num_workers=self.num_workers)
            
                    if self.mp:
                        self.model.module.train(False)  # Set model to evaluate mode
                        self.model.module.R.d_rate(0)
                        
                    if not self.mp:
                        self.model.train(False)  # Set model to evaluate mode
                        self.model.R.d_rate(0)

                # iterate batch
                for data in dataLoader:
                    # get the inputs
                    inputs_bt, labels_bt = data #<class 'torch.FloatTensor'> <class 'torch.LongTensor'>
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    # forward
                    outputs = self.model(Variable(inputs_bt.cuda()))
                    _, preds_bt = torch.max(outputs.data, 1)
                    preds_bt = preds_bt.cpu().view(-1)

                    # calsulate the loss
                    loss = self.criterion(outputs, Variable(labels_bt.cuda()))
                    
                    # backward + optimize only if in training phase
                    if phase == 'tra': 
                        loss.backward()
                        self.optimizer.step()
                        
                    # statistics
                    running_loss += loss.data[0]
                    N_T += torch.sum(preds_bt == labels_bt)
                    N_A += len(labels_bt)
                    for i in range(len(labels_bt)): accMat[labels_bt[i],preds_bt[i]] += 1
                        
                # record the performance
                mat = normalize(accMat.astype(np.float64),axis=1,norm='l1')
                K.matrixPlot(mat,self.dst + 'epoch/', phase + str(epoch))
                
                epoch_acc = np.trace(mat)
                epoch_loss = running_loss / N_A
                # epoch_acc = N_T / N_A
                
                self.record[phase].append((epoch, epoch_loss, epoch_acc))
                
                if type(epoch_loss) != float: epoch_loss = epoch_loss[0]
                print('{:5}:\n Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))    
                
                if phase == 'val':
                    # deep copy the model
                    if epoch_acc > self.best_acc:
                        self.best_acc = epoch_acc
                        self.best_epoch = epoch
                        self.best_model = copy.deepcopy(self.model)
                        torch.save(self.best_model, self.dst + 'model.pth')
                    
        
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f} in epoch: {}'.format(self.best_acc,self.best_epoch))
        torch.save(self.record, self.dst + str(self.best_epoch) + 'record.pth')
        return

   

    
    
    