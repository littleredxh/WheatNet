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

import KNet_utils as K
from KNet import KNet, KLoss, KSampler

PHASE = ['train', 'val']
RGBmean, RGBstdv = [0.429, 0.495, 0.259], [0.218, 0.224, 0.171]  

def normARG(minV, maxV, array):
    N = array.shape[0]
    min_num = array.min(0)
    array = array-min_num
    max_num = array.max(0)
    return array/max_num*(maxV-minV)+minV

class learn():
    def __init__(self, src, dst, gpuid):
        self.src = src
        self.dst = dst
        self.gpuid = gpuid
        
        if len(gpuid)>1: 
            self.mp = True
        else:
            self.mp = False

        self.batch_size = 20
        self.num_workers = 20
        self.init_lr = 0.01
        self.lr_decay_epoch = 5
        self.num_features = 11
        self.criterion = KLoss()
        self.record = {p:[] for p in PHASE}
        
        
    def run(self):
        if not self.setsys(): return
        self.loadData()
        self.setModel()
        self.printInfo()
        num_epochs=self.lr_decay_epoch*4
        self.train(num_epochs)
                                              
    def setsys(self):
        if not os.path.exists(self.src): print('src folder not exited'); return False
        if not torch.cuda.is_available(): print('No GPU detected'); return False
        if not os.path.exists(self.dst): os.makedirs(self.dst)
        torch.cuda.set_device(self.gpuid[0]); print('Current device is GPU: {}'.format(torch.cuda.current_device()))
        return True   
    
    def loadData(self):
        data_transforms = {'train': transforms.Compose([
                                    transforms.Scale(size=224*4, interpolation=Image.BICUBIC),
                                    transforms.RandomCrop(224*3),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(RGBmean, RGBstdv)]),
                             'val': transforms.Compose([
                                    transforms.Scale(size=224*4, interpolation=Image.BICUBIC),
                                    transforms.CenterCrop(224*3),
                                    transforms.ToTensor(),
                                    transforms.Normalize(RGBmean, RGBstdv)])}
        
        self.dsets = {p: datasets.ImageFolder(os.path.join(self.src, p), data_transforms[p]) for p in PHASE}
        self.class2indx = self.dsets['train'].class_to_idx
        self.indx2class = {v: k for k,v in self.class2indx.items()}
        self.class_size = {p: {k: 0 for k in self.class2indx} for p in PHASE }# number of images in each class
        self.N_classes = len(self.class2indx)# total number of classes
        self.bookmark = {p:[] for p in PHASE}# index bookmark
        torch.save(self.indx2class, self.dst+'indx2class.pth')
        
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
                print('--')
                print(idx)
                print(self.dsets[phase][sta][1])
                print(self.dsets[phase][end-1][1])
                try:
                    print(self.dsets[phase][end][1])
                except:
                    print('end')
                sta += self.class_size[phase][classkey]
        return

    def setModel(self):
        # create whole model
        Kmodel = KNet(self.num_features,self.N_classes)
        
        # parallel computing and opt setting
        if self.mp:
            print('Training on Multi-GPU')
            self.batch_size = self.batch_size*len(self.gpuid)
            self.model = torch.nn.DataParallel(Kmodel,device_ids=self.gpuid).cuda()#
            print(self.model)
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

    def DataLoaders(self):
        self.sampler = {PHASE[0]:KSampler(self.bookmark[PHASE[0]]),PHASE[1]:KSampler(self.bookmark[PHASE[1]],balance=False)}
        self.dataLoader = {p: torch.utils.data.DataLoader(self.dsets[p], batch_size=self.batch_size, sampler=self.sampler[p], num_workers=self.num_workers, drop_last = True) for p in PHASE}
        return
    
    def lr_scheduler(self, epoch):
        lr = self.init_lr * (0.2**(epoch // self.lr_decay_epoch))
        if epoch % self.lr_decay_epoch == 0: print('LR is set to {}'.format(lr))
        for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        return

    def train(self, num_epochs):
        # recording time and epoch acc and best result
        since = time.time()
        self.best_tra = 0.0
        self.best_epoch = 0
        for epoch in range(num_epochs):
            self.DataLoaders()
            print('Epoch {}/{} \n '.format(epoch, num_epochs - 1) + '-' * 40)
            
            for phase in PHASE:
                # recording the result
                accMat = np.zeros((self.N_classes,self.N_classes))
                running_loss = 0.0
                N_T, N_A = 0,0
                
                # Adjust the model for different phase
                if phase == 'train':
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
                        
                if phase == 'val':
                    if self.mp:
                        self.model.module.train(False)  # Set model to evaluate mode
                        self.model.module.R.d_rate(0)
                        
                    if not self.mp:
                        self.model.train(False)  # Set model to evaluate mode
                        self.model.R.d_rate(0)

                # iterate batch
                for data in self.dataLoader[phase]:
                    # get the inputs
                    inputs_bt, labels_bt = data #<class 'torch.FloatTensor'> <class 'torch.LongTensor'>
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    # forward
                    outputs = self.model(Variable(inputs_bt.cuda()))
                    _, preds_bt = torch.max(outputs.data, 1)
                    preds_bt = preds_bt.cpu().view(-1)

                    # calsulate the loss
                    loss = self.criterion(outputs, labels_bt)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train': 
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
                
                epoch_tra = np.trace(mat)
                epoch_loss = running_loss / N_A
                epoch_acc = N_T / N_A
                
                self.record[phase].append((epoch, epoch_loss, epoch_acc))
                
                if type(epoch_loss) != float: epoch_loss = epoch_loss[0]
                print('{:5}:\n Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))    
                
                
                if phase == 'val':
                    # calculate dynamic weight
                    # maxV = (0.4*((num_epochs-epoch)/num_epochs)+1)
                    # w = normARG(1, maxV, (2-mat.diagonal())).tolist()
                    # self.criterion = KLoss(w)
                    # print(["%.2f" % a for a in w])
                    
                    # deep copy the model
                    if epoch_tra > self.best_tra and epoch > num_epochs/2:
                        self.best_tra = epoch_tra
                        self.best_epoch = epoch
                        self.best_model = copy.deepcopy(self.model)
                        torch.save(self.best_model, self.dst + 'model.pth')
                    
        
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f} in epoch: {}'.format(self.best_tra,self.best_epoch))
        torch.save(self.record, self.dst + str(self.best_epoch) + 'record.pth')
        K.recordPlot(self.record, self.dst)
        return
    
    def view(self):
        K.folderViewL(self.src,self.dst+'montage/')
   

    
    
    