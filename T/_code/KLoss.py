class KSampler(Sampler):
    def __init__(self,bookmark,balance=True):
        if balance==False:
            self.idx = np.arange(bookmark[0][0],bookmark[-1][1]).tolist()
            return
        
        self.bookmark = bookmark
        interval_list = []
        sp_list_sp = []
        
        # find the max interval
        len_max = 0
        for b in bookmark:
            interval_list.append(np.arange(b[0],b[1]))
            if b[1]-b[0] > len_max: len_max = b[1]-b[0]
            
        for l in interval_list:
            if l.shape[0]<len_max:
                l_ext = np.random.choice(l,len_max-l.shape[0])
                l_ext = np.concatenate((l, l_ext), axis=0)
                l_ext = np.random.permutation(l_ext)
            else:
                l_ext = np.random.permutation(l)
            sp_list_sp.append(l_ext)

        self.idx = np.vstack(sp_list_sp).T.reshape((1,-1)).flatten().tolist()
            
    def __iter__(self):
        return iter(self.idx)
    def __len__(self):
        return len(self.idx)
    
class KLoss(nn.Module):
    def __init__(self,w=None):
        super(KLoss, self).__init__()
        self.w = w
        self.sfmx = nn.Softmax()
        
    def normLM(self,mat):# input N by F
        F = mat.size(1)
        maxw, _ = mat.max(1)
        w = torch.t(maxw.repeat(F,1)).contiguous()
        return w
    
    def normL1(self,mat):# input N by F
        F = mat.size(1)
        w = torch.t(mat.sum(1).repeat(F,1)).contiguous()
        return mat.div(w)  
    
    def label2dist(self,target):
        target_mat = torch.ones(self.N,self.C)*0.01
        for i in range(self.N):
            idx = target[i]
            target_mat[i][idx] = 0.8
            if idx-1 >= 0:     target_mat[i][idx-1] = 0.15
            if idx+1 < self.C: target_mat[i][idx+1] = 0.15
            if idx-2 >= 0:     target_mat[i][idx-1] = 0.075
            if idx+2 < self.C: target_mat[i][idx+1] = 0.075
                
        return Variable(self.normL1(target_mat)).cuda()
    
    def weight(self,target):
        weight_Vec = torch.zeros(self.N)
        for i in range(self.N):
            weight_Vec[i] = self.w[target[i]]
        return Variable(weight_Vec.cuda())
    
    def forward(self, predict, target):
        # predict: N by C
        # target: N
        self.N,self.C = predict.size()
        # turn label vector into N by C
        target_mat = self.label2dist(target)
        target_w = self.normLM(target_mat)
        if self.w==None:
            loss = (target_mat - predict).abs().sum(1).sum(0)
            # loss = (target_mat - predict).abs().mul(target_w).sum(1).sum(0)
        else:
            loss = (target_mat - predict).abs().sum(1).dot(self.weight(target))
            # loss = (target_mat - predict).abs().mul(target_w).sum(1).dot(self.weight(target))
        return loss