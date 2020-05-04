# -*- coding: utf-8 -*-
'''
Created on 2020年2月20日

@author: xmblc
'''
import numpy as np
import netCDF4 as nc
import numpy.ma as ma
from SQL import psySQL
import sys
import os
import calendar

def getSHH_mean(filepath,year,month,areaScope,prenum):
    #获取卫星遥感海面高度月平均
    dirs = 'ssha' + ' ' + str(year) + ' ' + '0.25'
    allfilepath = os.path.join(filepath,'SSH',dirs)
    all_name = all_files_in_path(allfilepath, '.nc')
    monthRange = calendar.monthrange(year,month)
    num = monthRange[1]
    names = all_name[prenum:prenum+num]
    
    count = 0
    for name in names:
        name = name+'.nc'
        file = os.path.join(allfilepath,name)
        res = read_satellite_datas(areaScope, file, features = ['sla'], grid_size=0.25, depth = 0)
        if count == 0:
            values = res['sla']
        else:
            values += res['sla']
        count+=1
    meanValues = values/count
    prenum += num
    return meanValues,prenum

def getSST_mean(filepath,year,month,areaScope,prenum):
    #获取卫星遥感海面温度月平均
    allfilepath = os.path.join(filepath,'SST',str(year),'AVHRR')
    all_name = all_files_in_path(allfilepath, '.nc')
    monthRange = calendar.monthrange(year,month)
    num = monthRange[1]
    names = all_name[prenum:prenum+num]
    
    count = 0
    for name in names:
        name = name+'.nc'
        file = os.path.join(allfilepath,name)
        res = read_satellite_datas(areaScope, file, features = ['sst'], grid_size=0.25, depth = 0)
        if count == 0:
            values = res['sst']
        else:
            values += res['sst']
        count+=1
    meanValues = values/count
    prenum += num
    return meanValues,prenum

def getArgo_to_SQL(areaScope,filepath,year,month,prenum,id_count):
    code_dict = {}
    allfilepath = os.path.join(filepath,'ARGO','2016')
    all_name = all_files_in_path(allfilepath, '.nc')
    name = all_name[month-1]+'.nc'
    file = os.path.join(allfilepath,name)
    
    res = read_argo_data(areaScope, file, features=['salt','temp','MLD','ILD'], grid_size=1, depth = 0)
    temp = res['temp']
    salt = res['salt']
    mixty = res['MLD']
    isotm = res['ILD']
    
    meanSSH,SSH_prenum = getSHH_mean(filepath,year,month,areaScope,prenum)
    meanSST,SST_prenum = getSST_mean(filepath,year,month,areaScope,prenum)
    
    minLat = float(areaScope['minLat'])
    maxLat = float(areaScope['maxLat'])
    minLon = float(areaScope['minLon'])
    maxLon = float(areaScope['maxLon'])
    latSteps = int((maxLat - minLat)/1 +0.1) + 1
    lonSteps = int((maxLon - minLon)/1 +0.1) + 1
    for x in range(latSteps):
        for y in range(lonSteps):
            lat = minLat + x*1
            lon = minLon + y*1
            SSH_1 = meanSSH[4*x,4*y]
            SSH_2 = meanSSH[4*x,4*y+1]
            SSH_3 = meanSSH[4*x+1,4*y]
            SSH_4 = meanSSH[4*x+1,4*y+1]
            SSH_m = (SSH_1+SSH_2+SSH_3+SSH_4)/4
            
            SST_1 = meanSST[4*x,4*y]
            SST_2 = meanSST[4*x,4*y+1]
            SST_3 = meanSST[4*x+1,4*y]
            SST_4 = meanSST[4*x+1,4*y+1]
            SST_m = (SST_1+SST_2+SST_3+SST_4)/4
            
            ARGO_t = temp[:,x,y]
            ARGO_s = salt[:,x,y]
            ARGO_m = mixty[x,y]
            ARGO_i = isotm[x,y]
            
            pres_code = 0
            for pres in pres_list:
                code_dict['pres'] = pres
                code_dict['id']  = id_count
                code_dict['lat'] = lat
                code_dict['lon'] = lon
                code_dict['temp'] = ARGO_t[pres_code]
                code_dict['salt'] = ARGO_s[pres_code]
                code_dict['mld'] = ARGO_m
                code_dict['ild'] = ARGO_i
                code_dict['SSH'] = SSH_m
                code_dict['SST'] = SST_m
                code_dict['year'] = year
                code_dict['month'] = month
                sSQL = psySQL.selectOperate(id_count)
                if sSQL == None:
                    psySQL.insertOperate(code_dict)
                    id_count+=1
                else:
                    continue
                pres_code+=1
                print(id_count)
                print('lat:'+str(lat)+' lon:'+str(lon))
                
    return SSH_prenum,SST_prenum,id_count
            
            
            


def all_files_in_path(file_dir, filetype):   
    all_name=[]
    for _, _, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == filetype:  
                all_name.append(os.path.splitext(file)[0])
    return all_name

def read_satellite_datas(areaScope, filename, features, grid_size, depth = 0):
    '''
                读取卫星遥感数据，传入文件完整路径名，
    '''
    # Lon值域[0.125,359.875],Lat值域[-89.875,89.875]，且两个经度、两个纬度值不允许一致，且数值精确到小数点后1位，调取函数之前需检查判断数据合理性
    regionaldatas = {}
    try:        
        os.chdir(os.path.dirname(filename))
        data = nc.Dataset(os.path.basename(filename))
        if features == ['sla']:
            ncStLat = np.array(data.variables['latitude'])[0]
            ncStLon = np.array(data.variables['longitude'])[0] 
            ncEdLat = np.array(data.variables['latitude'])[-1]
            ncEdLon = np.array(data.variables['longitude'])[-1] 
        else:
            ncStLat = np.array(data.variables['lat'])[0]
            ncStLon = np.array(data.variables['lon'])[0] 
            ncEdLat = np.array(data.variables['lat'])[-1]
            ncEdLon = np.array(data.variables['lon'])[-1]  
        for feature in features:
            if feature not in data.variables:
                continue
            
            if 'global' in areaScope:                   #读取NC文件全部数据
                startLat = ncStLat
                endLat = ncEdLat
                startLon = ncStLon
                endLon = ncEdLon

            elif len(areaScope) == 4:
                startLat = float(areaScope['minLat'])
                endLat = float(areaScope['maxLat'])
                startLon = float(areaScope['minLon'])
                endLon = float(areaScope['maxLon'])
                
                if startLon < 0:
                    startLon = startLon + 360
                if endLon < 0:
                    endLon = endLon + 360
                    
                if startLon > endLon:                                       # 跨日经线的情况
                    regionaldata_1 = data.variables[feature][:, :, int((startLat-ncStLat)/grid_size+0.1):int((endLat-ncStLat)/grid_size + 0.1)+1, int((startLon -ncStLon)/grid_size +0.1):int((360-ncStLon)/grid_size +0.1)+1]
                    regionaldata_2 = data.variables[feature][:, :, int((startLat-ncStLat)/grid_size+0.1):int((endLat-ncStLat)/grid_size + 0.1)+1, int((ncStLon-ncStLon)/grid_size+0.1):int((endLon -ncStLon)/grid_size +0.1)+1] 
                    regionaldata=np.concatenate((regionaldata_1,regionaldata_2),axis=2)
                    regionaldata = np.array(regionaldata) 
                else:
                    if feature=='sla':
                        regionaldata = data.variables[feature][:, int((startLat-ncStLat)/grid_size+0.1):int((endLat-ncStLat)/grid_size + 0.1)+2, int((startLon -ncStLon)/grid_size +0.1):int((endLon -ncStLon)/grid_size +0.1)+2]
                        regionaldata = regionaldata[0]
                    else:
                        regionaldata = data.variables[feature][:,:, int((startLat-ncStLat)/grid_size+0.1):int((endLat-ncStLat)/grid_size + 0.1)+2, int((startLon -ncStLon)/grid_size +0.1):int((endLon -ncStLon)/grid_size +0.1)+2]
                        regionaldata = regionaldata[0][0]
            
            elif len(areaScope) == 2:
                Lat = float(areaScope['lat'])
                Lon = float(areaScope['lon'])
                if Lon < 0:
                    Lon = Lon + 360
                if feature=='sla':
                    regionaldata = data.variables[feature][:, int((Lat-ncStLat)/grid_size+0.1), int((Lon -ncStLon)/grid_size +0.1)]
                    
                elif feature=='sst':
                    regionaldata = data.variables[feature][:, :, int((Lat-ncStLat)/grid_size+0.1), int((Lon -ncStLon)/grid_size +0.1)]
                    regionaldata = regionaldata[0][0]
            regionaldatas[feature] = regionaldata
        return regionaldatas
    except Exception as e:  
        print("执行 %s 函数发生错误：%s"  % (sys._getframe().f_code.co_name, e))
        return None
    
def read_argo_data(areaScope, filename, features, grid_size, depth = 0):
    '''
         读取argo数据
    '''
    regionaldatas = {}
    try:        
        os.chdir(os.path.dirname(filename))
        data = nc.Dataset(os.path.basename(filename))
        ncStLat = np.array(data.variables['lat'])[0]
        ncStLon = np.array(data.variables['lon'])[0] 
        ncEdLat = np.array(data.variables['lat'])[-1]
        ncEdLon = np.array(data.variables['lon'])[-1] 
        
        for feature in features:
            if feature not in data.variables:
                continue
            
            if 'global' in areaScope:                   #读取NC文件全部数据
                startLat = ncStLat
                endLat = ncEdLat
                startLon = ncStLon
                endLon = ncEdLon

            elif len(areaScope) == 4:
                startLat = float(areaScope['minLat'])
                endLat = float(areaScope['maxLat'])
                startLon = float(areaScope['minLon'])
                endLon = float(areaScope['maxLon'])
                
                if startLon < 0:
                    startLon = startLon + 360
                if endLon < 0:
                    endLon = endLon + 360
                    
                if startLon > endLon:                                       # 跨日经线的情况
                    regionaldata_1 = data.variables[feature][:, :, int((startLat-ncStLat)/grid_size+0.1):int((endLat-ncStLat)/grid_size + 0.1)+1, int((startLon -ncStLon)/grid_size +0.1):int((360-ncStLon)/grid_size +0.1)+1]
                    regionaldata_2 = data.variables[feature][:, :, int((startLat-ncStLat)/grid_size+0.1):int((endLat-ncStLat)/grid_size + 0.1)+1, int((ncStLon-ncStLon)/grid_size+0.1):int((endLon -ncStLon)/grid_size +0.1)+1] 
                    regionaldata=np.concatenate((regionaldata_1,regionaldata_2),axis=2)
                    regionaldata = np.array(regionaldata) 
                else:
                    if feature=='MLD' or feature=='ILD':
                        regionaldata = data.variables[feature][:,int((startLat-ncStLat)/grid_size+0.1):int((endLat-ncStLat)/grid_size + 0.1)+1, int((startLon -ncStLon)/grid_size +0.1):int((endLon -ncStLon)/grid_size +0.1)+1]
                        regionaldata = regionaldata[0]
                    else:
                        regionaldata = data.variables[feature][:, :, int((startLat-ncStLat)/grid_size+0.1):int((endLat-ncStLat)/grid_size + 0.1)+1, int((startLon -ncStLon)/grid_size +0.1):int((endLon -ncStLon)/grid_size +0.1)+1]
                        regionaldata = regionaldata[0]
            regionaldatas[feature] = regionaldata
        return regionaldatas
    
    except Exception as e:  
        print("执行 %s 函数发生错误：%s"  % (sys._getframe().f_code.co_name, e))
        return None


def read_argo_point(point, filename, features, grid_size, depth=0):
    '''
         读取argo数据
    '''
    regionaldatas = {}
    try:
        os.chdir(os.path.dirname(filename))
        data = nc.Dataset(os.path.basename(filename))
        ncStLat = np.array(data.variables['lat'])[0]
        ncStLon = np.array(data.variables['lon'])[0]
        ncEdLat = np.array(data.variables['lat'])[-1]
        ncEdLon = np.array(data.variables['lon'])[-1]

        for feature in features:
            if feature not in data.variables:
                continue

            Lat = point['lat']
            Lon = point['lon']

            if Lon< 0:
                Lon = Lon + 360
            if Lon < 0:
                Lon = Lon + 360

            regionaldata = data.variables[feature][:, int((Lat - ncStLat) / grid_size + 0.1), int((Lon - ncStLon) / grid_size + 0.1)]
            regionaldata = regionaldata[0]
            regionaldatas[feature] = regionaldata
        return regionaldatas

    except Exception as e:
        print("执行 %s 函数发生错误：%s" % (sys._getframe().f_code.co_name, e))
        return None

def getMean(point, filename, features, grid_size):

    value = read_argo_point(point, filename, features, grid_size, depth=0)


'''  
pres_list = [0,5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1400,1500,1600,1700,1800,1900,1975]
areaScope = {'minLat':20.5,'maxLat':40.5,'minLon':120.5,'maxLon':140.5}
year = 2017
filepath = 'D:\Datas'
prenum = id_count = 0
psySQL.connectPostgreSQL()
for month in range(0,12):
    month+=1
    SSH_prenum,SST_prenum,id_count = getArgo_to_SQL(areaScope,filepath,year,month,prenum,id_count)
    if SSH_prenum == SST_prenum:
        prenum = SSH_prenum   
'''
    
    
    