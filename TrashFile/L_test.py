import KNet_keyfile
csvfiles = KNet_keyfile.csvfiles_pctL

import KNet_TimeSeriesPct
src = '/project/focus/hong/AllData/'
dst = 'TS_result6L_17/'
model_path = 'pct6_result/result13_paper/model.pth'

ans = KNet_TimeSeriesPct.csv2pred(src,dst,csvfiles,model_path)
ans.run()