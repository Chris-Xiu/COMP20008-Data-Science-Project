import pandas as pd
cvd1 = pd.read_csv('1.csv',encoding = 'ISO-8859-1')
cvd2 = pd.read_csv('2.csv')
cp2=cvd2.copy()
cp2['num_facility']=[1]*len(copy2[' numberfieldcourts'])
final=cp2.groupby(' lga').agg({'num_facility':sum})
final2=final[final['num_facility']>1]