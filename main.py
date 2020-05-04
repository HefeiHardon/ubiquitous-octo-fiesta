# -*- coding: utf-8 -*-
from SQL import psySQL
import numpy as np
from neural_network import def_models,draw_curve
from data_pre import argo
from data_pre import data_helpers
import tensorflow as tf
import random as rd
import keras
import os
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model

#连接数据库，读取1W个数据
areaScope={}
areaScope['minlat']=20
areaScope['maxlat']=30
areaScope['minlon']=120
areaScope['maxlon']=130
rows1 = psySQL.selectOperate(id_count=2000,month=4,year=2016)
#rows2 = psySQL.selectOperate(id_count=1000,month=2,year=2016)
#rows3 = psySQL.selectOperate(id_count=1000,month=3,year=2016)
'''
rows4 = psySQL.selectOperate(id_count=1000,month=4,year=2016)
rows5 = psySQL.selectOperate(id_count=1000,month=5,year=2016)
rows6 = psySQL.selectOperate(id_count=1000,month=6,year=2016)
rows7 = psySQL.selectOperate(id_count=1000,month=7,year=2016)
rows8 = psySQL.selectOperate(id_count=1000,month=8,year=2016)
rows9 = psySQL.selectOperate(id_count=1000,month=9,year=2016)
rows10 = psySQL.selectOperate(id_count=1000,month=10,year=2016)
rows11 = psySQL.selectOperate(id_count=1000,month=11,year=2016)
rows12 = psySQL.selectOperate(id_count=1000,month=12,year=2016)
'''
rows13 = psySQL.selectOperate2017(id_count=2000,month=4,year=2017)

#rows14 = psySQL.selectOperate2017(id_count=1000,month=2,year=2017)
#rows15 = psySQL.selectOperate2017(id_count=1000,month=3,year=2017)
'''
rows16 = psySQL.selectOperate2017(id_count=1000,month=4,year=2017)
rows17 = psySQL.selectOperate2017(id_count=1000,month=5,year=2017)
rows18 = psySQL.selectOperate2017(id_count=1000,month=6,year=2017)
rows19 = psySQL.selectOperate2017(id_count=1000,month=7,year=2017)
rows20 = psySQL.selectOperate2017(id_count=1000,month=8,year=2017)
rows21 = psySQL.selectOperate2017(id_count=1000,month=9,year=2017)
rows22 = psySQL.selectOperate2017(id_count=1000,month=10,year=2017)
rows23 = psySQL.selectOperate2017(id_count=1000,month=11,year=2017)
rows24 = psySQL.selectOperate2017(id_count=1000,month=12,year=2017)
'''
#rows1 = psySQL.selectOperate(1500,type='scope',areaScope=areaScope,month=1)#rows为列表
#rows2 = psySQL.selectOperate(1500,type='scope',areaScope=areaScope,month=2)#rows为列表
#rows3 = psySQL.selectOperate(1500,type='scope',areaScope=areaScope,month=3)#rows为列表
#rows4 = psySQL.selectOperate(20,type='scope',areaScope=areaScope,month=4)#rows为列表
#rows5 = psySQL.selectOperate(20,type='scope',areaScope=areaScope,month=5)#rows为列表
#rows6 = psySQL.selectOperate(20,type='scope',areaScope=areaScope,month=6)#rows为列表
#rows7 = psySQL.selectOperate(20,type='scope',areaScope=areaScope,month=7)#rows为列表
#rows8 = psySQL.selectOperate(1000,type='scope',areaScope=areaScope,month=8)#rows为列表
#rows9 = psySQL.selectOperate(1000,type='scope',areaScope=areaScope,month=9)#rows为列表
#rows10 = psySQL.selectOperate(20,type='scope',areaScope=areaScope,month=10)#rows为列表
#rows11 = psySQL.selectOperate(20,type='scope',areaScope=areaScope,month=11)#rows为列表
#rows12 = psySQL.selectOperate(20,type='scope',areaScope=areaScope,month=12)#rows为列表
#rows13 = psySQL.selectOperate(20,type='scope',areaScope=areaScope,month=1)#rows为列表
#rows14 = psySQL.selectOperate(20,type='scope',areaScope=areaScope,month=2)#rows为列表
#rows15 = psySQL.selectOperate(20,type='scope',areaScope=areaScope,month=3)#rows为列表
#rows16 = psySQL.selectOperate(20,type='scope',areaScope=areaScope,month=4)#rows为列表
#rows17 = psySQL.selectOperate(20,type='scope',areaScope=areaScope,month=5)#rows为列表
#rows18 = psySQL.selectOperate(20,type='scope',areaScope=areaScope,month=6)#rows为列表
#rows19 = psySQL.selectOperate(20,type='scope',areaScope=areaScope,month=7)#rows为列表
#rows20 = psySQL.selectOperate(20,type='scope',areaScope=areaScope,month=8)#rows为列表
#rows21 = psySQL.selectOperate(20,type='scope',areaScope=areaScope,month=9)#rows为列表
#rows22 = psySQL.selectOperate(20,type='scope',areaScope=areaScope,month=10)#rows为列表
#rows23 = psySQL.selectOperate(20,type='scope',areaScope=areaScope,month=11)#rows为列表
#rows24 = psySQL.selectOperate(20,type='scope',areaScope=areaScope,month=12)#rows为列表
rows = rows1 +rows13#+ rows2 + rows3 +  rows13 + rows14 + rows15#rows4 + rows5 + rows6 + rows7 + rows8 + rows9 + rows10 + rows11 + rows12 + rows13 + rows14 + rows15 + rows16 + rows17 + rows18 + rows19 + rows20 + rows21 + rows22 + rows23 + rows24
#rows = rows1 #+ rows2+ rows3+ rows4+ rows5+ rows6
rows_array = np.array(rows)
#将无效值-9999掩膜处理
mask_array = rows_array == 9999
rows_array = np.ma.array(rows_array,mask=mask_array)
shape = rows_array.shape

#获取WOD数据
values = psySQL.selectOperate(id_count=200,type='equal',areaScope=areaScope,month=4,year=2016)
values = np.array(values)
v_lat = values[:,1].astype(float)
v_lon = values[:,2].astype(float)
v_depth = values[:,3].astype(float)
v_temp = values[:,4].astype(float)
v_salt = values[:,5].astype(float)
v_ssh = values[:,8].astype(float)
v_sst = values[:,9].astype(float)
v_month = values[:,10].astype(float)
v_year = values[:,-1].astype(float)
point={}
point['lat']=v_lat[0]
point['lon']=v_lon[0]
month=float(v_month[0])
v_month =np.array([month]*len(v_lat))

all_name = argo.all_files_in_path('D:\Datas\ARGO', '.nc')
name = all_name[int(month)-1]+'.nc'
filename = os.path.join('D:\Datas\ARGO',str(2016),name)
res = argo.read_argo_point(point, filename, features=['MLD','ILD'], grid_size=1, depth=0)
v_mld = np.array([res['MLD']]*57)
v_ild = np.array([res['ILD']]*57)
pres_list = np.array([0,5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1400,1500,1600,1700,1800,1900,1975])
v_pres=pres_list[:len(v_lat)]

#pre_array = all_data[0:56,:]


#分组取出各特征对应的数据
lats = rows_array[:,1].astype(float)
nums = len(lats)
lats = np.append(lats,v_lat)
lons = rows_array[:,2].astype(float)
lons = np.append(lons,v_lon)
pres = rows_array[:,3].astype(float)
pres = np.append(pres,v_pres)
temp = rows_array[:,4].astype(float)
temp = np.append(temp,v_temp)
salt = rows_array[:,5].astype(float)
salt = np.append(salt,v_salt)
mld = rows_array[:,6].astype(float)
mld = np.append(mld,v_mld)
ild = rows_array[:,7].astype(float)
ild = np.append(ild,v_ild)
ssh = rows_array[:,8].astype(float)
ssh = np.append(ssh,v_ssh)
sst = rows_array[:,9].astype(float)
sst = np.append(sst,v_sst)
month = rows_array[:,10].astype(float)
month = np.append(month,v_month)
#day = rows_array[:,9].astype(float)
year = rows_array[:,11].astype(float)
year = np.append(year,v_year)

lats_n = def_models.data_standardization(lats)
lons_n = def_models.data_standardization(lons)
pres_n = def_models.data_standardization(pres)
temp_n = def_models.data_standardization(temp)
salt_n = def_models.data_standardization(salt)
mld_n = def_models.data_standardization(mld)
ild_n = def_models.data_standardization(ild)
ssh_n = def_models.data_standardization(ssh)
sst_n = def_models.data_standardization(sst)
month_n = def_models.data_standardization(month)
#day = def_models.data_standardization(day)
year_n = def_models.data_standardization(year)
pre_array=np.concatenate((lats_n[nums:nums+56][None],lons_n[nums:nums+56][None],pres_n[nums:nums+56][None],salt_n[nums:nums+56][None],mld_n[nums:nums+56][None],ild_n[nums:nums+56][None],ssh_n[nums:nums+56][None],sst_n[nums:nums+56][None],month_n[nums:nums+56][None],year_n[nums:nums+56][None]),axis=0)
all_data = np.concatenate((lats_n[:nums],lons_n[:nums],pres_n[:nums],salt_n[:nums],mld_n[:nums],ild_n[:nums],ssh_n[:nums],sst_n[:nums],month_n[:nums],year_n[:nums]),axis=0)
all_data = all_data.reshape(10,nums)
all_data = all_data.T
#再掩膜处理
mask_array1 = all_data == 9999
all_data = np.ma.array(all_data,mask=mask_array1)
mask_array2 = temp == 9999
temp = np.ma.array(temp,mask=mask_array2)
train_temp = temp[:nums]
test_temp = temp[nums:]

#选取单点有20个坡面温度数据的点
'''
depth_list = list(depth)
count = start_index = end_index = 0
for depth_value in depth_list:
    if count == 0:
        start_depth = pre_depth = depth_value
    else:
        if end_index - start_index > 20:
            break
        elif depth_value <= pre_depth:
            pre_depth = start_depth = depth_value
            start_index = end_index = count
        else:
            pre_depth = depth_value
            end_index += 1
    count += 1
'''
#np.random.shuffle(all_data)
train_data = all_data[:int(shape[0]*0.8),:]
test_data =all_data[int(shape[0]*0.8):,:]
train_targets = train_temp[:int(shape[0]*0.8)]
test_targets = train_temp[int(shape[0]*0.8):]
pre_array = train_data[:56,:]

#限制GPU使用率
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
#pre_array = (np.array([train_data[0]])).T
#pre_array = all_data[0:56,:]
history,model = def_models.train_model(train_data,train_targets)

model.fit(train_data,train_targets)
mae_history = history.history['val_mae']
epochs = mae_history.index(min(mae_history))
print(np.mean(np.array(mae_history)))
#model = load_model('wod_model.h5')
#model.fit(train_data,train_targets,epochs=220,batch_size=16,verbose=1)
test_mse_score,test_mae_score = model.evaluate(test_data,test_targets)
print(test_mae_score)


test_predictions = model.predict(pre_array).flatten()
#plt.scatter(test_targets,test_predictions)
print(np.mean(test_predictions-test_temp[:56]))
print(test_predictions-test_temp[:56])

x1=train_temp[:50]
y1=v_depth[:50]
x2=test_predictions[:50]
y2=pres[:50]
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x1,y1,'g-')
ax2.plot(x2,y2,'b-')
print(x1)
print(y1)
print(x2)
print(y2)

ax1.set_xlabel('Tempretures')
ax1.set_ylabel('depth value',color='g')
ax2.set_ylabel('pressure value',color='r')
plt.show()

'''
plt.plot(test_temp[0:50],v_depth[:50],label='True Values')
plt.plot(test_predictions[:50],pres[:50],label='Predictions')
plt.ylabel('Pressure Values')
plt.xlabel('Tempretures')
plt.legend()
#plt.axis('equal')
#plt.xlim(plt.xlim())
#plt.ylim(plt.ylim())
#_ = plt.plot([-100,100],[-100,100])
plt.show()
'''


#all_scores = def_models.K_fold_verfication(train_data,train_targets,test_data,test_targets,k=4,epochs=500)

#print(all_scores)
