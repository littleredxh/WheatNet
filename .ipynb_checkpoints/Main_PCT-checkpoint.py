from Train import learn
from keyfile import csvfiles_pct_A17, csvfiles_pct_A16
from DataPreparePct import csv2img
from Utils import cpImg

src = '/project/focus/hong/AllData/'
dst = '_result/'

# perpare the training images
csvfiles = {'17':csvfiles_pct_A17,'16':csvfiles_pct_A16}
data_dict = csv2img(src,dst,csvfiles).run()
    
# train process
learn(src, dst, data_dict).run()