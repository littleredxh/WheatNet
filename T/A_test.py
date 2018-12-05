import KNet_keyfile
csvfiles = KNet_keyfile.csvfiles_awns16

import KNet_TimeSeriesAwn
src = '/project/focus/hong/AllData/'
dst = 'awns2_result/'

model_path = 'awns2_result/0/'
# perpare the training images
ans = KNet_TimeSeriesAwn.csv2pred(src,dst,csvfiles,model_path)
ans.run()


