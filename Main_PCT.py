from Train import learn
from keyfile import csvfiles_pct_A17, csvfiles_pct_A16
from DataPreparePct import csv2img
import torch

src = '/project/focus/hong/datasets/KSU_wheat/'
dst = '_result/'

# perpare the training images
# csvfiles = {'17':csvfiles_pct_A17,'16':csvfiles_pct_A16}
# data_dict = csv2img(src,dst,csvfiles).run()
# torch.save(data_dict,'pct_data.pth')
data_dict = torch.load('pct_data.pth')
# train process
learn(src, dst, data_dict).run(10)