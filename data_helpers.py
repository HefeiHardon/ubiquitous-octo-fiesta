# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import netCDF4 as nc
import os
import datetime
import arrow
import sys
import numpy.ma as ma
from data_pre.dateFilesList import DaysList
from SQL import psySQL



class load_data_and_labels():
    '''
        加载数据并处理标签
    '''
    def __init__(self, timeStatInfo, Data_path):
        '''
        Constructor
        '''
        self.Data_path = Data_path
        self.timeStatTnfo = timeStatInfo
        self.timeStatType = timeStatInfo['timeStatType']
        self.statUnit = timeStatInfo['statUnit']
        
      #得到海面温度的卫星遥感资料数据
    def getSatelliteSST(self,point):
        #tem_read_all = []
        #dayscope = np.array([])
        
        #obj = DaysList(self.timeStatTnfo)
        #all_days_array = obj.AllDaysArray()
        #errCount = 0
        #numUnit = np.size(all_days_array, 0)
        #if self.statUnit == 1 or self.statUnit == 0:   #如果是月和年为单位，就是二维
            #for unit_list in range(numUnit):
                #daylist = all_days_array[unit_list][:]
                #dayscope = np.append(dayscope,daylist)
        #else:
            #dayscope = all_days_array
        #for day in dayscope:
        day = self.timeStatTnfo['endDay']
        year = day[:4]
        file = 'avhrr-only-v2.' + day + '.nc'
        filePath = os.path.join('SST', year, 'AVHRR')
        fileName = os.path.join(self.Data_path, filePath, file) # D:\Datas\SST\2016\AVHRR\...
        res = self.read_satellite_datas(point,fileName,features = ['sst'], grid_size=0.25, depth = 0)
        print(fileName)
        tem_read = res['sst']
            
        tem_read = np.array(tem_read)
        maskArray = tem_read == -999
        tem_read_all = ma.array(tem_read, mask=maskArray)                         # 矩阵掩码处理，去除无效值       
        return tem_read_all
    
    def get_all_SSH_filenames(self):
        minYear = self.timeStatTnfo['startDay'][:4]
        maxYear = self.timeStatTnfo['endDay'][:4]
        all_names=[]
        for year in range(int(minYear),int(maxYear)+1):
            dirs = 'ssha' + ' ' + str(year)[:4] + ' ' + '0.25'
            path_name = os.path.join(self.Data_path, 'SSH', dirs) # path_ssh='D:\Datas\SSH\ssha 2016 0.25'
            all_name = all_files_in_path(path_name, '.nc')
            all_names.append(all_name)
        return all_names,path_name
    
    #得到海面高度异常值的卫星遥感资料数据
    def getSatelliteSSH(self,point):
        ssh_read_all= []

        #obj = DaysList(self.timeStatType) 
        #all_days_array = obj.AllDaysArray()
        all_SSH_filenames,path_name = self.get_all_SSH_filenames()
        
        minYear=self.timeStatTnfo['startDay'][:4]
        maxYear=self.timeStatTnfo['endDay'][:4]
        minMonth=self.timeStatTnfo['startDay'][4:6]
        maxMonth=self.timeStatTnfo['endDay'][4:6]
        minDay=self.timeStatTnfo['startDay'][6:8]
        maxDay=self.timeStatTnfo['endDay'][6:8]
        date1 = datetime.date(year=int(minYear), month=int(minMonth), day=int(minDay))
        date2 = datetime.date(year=int(maxYear),month=int(maxMonth),day=int(maxDay))
        nums = (date2-date1).days
        
        file = all_SSH_filenames[int(maxYear)-int(minYear)][nums]
        
        filename = os.path.join(path_name,file+'.nc')
        print(filename)
        res = self.read_satellite_datas(point, filename, features = ['sla'], grid_size=0.25, depth = 0)
        if res == None:
            ssh_read = None
        else:
            ssh_read = res['sla']
        ssh_read = np.array(ssh_read)
        maskArray = ssh_read == -2147483647
        ssh_read = ma.array(ssh_read, mask=maskArray)                         # 矩阵掩码处理，去除无效值       
        return ssh_read
    
        
    def read_satellite_datas(self, areaScope, filename, features, grid_size, depth = 0):
        '''
                    读取卫星遥感数据，传入文件完整路径名，
        '''
        # Lon值域[0.125,359.875],Lat值域[-89.875,89.875]，且两个经度、两个纬度值不允许一致，且数值精确到小数点后1位，调取函数之前需检查判断数据合理性
        regionaldatas = {}
        try:        
            os.chdir(os.path.dirname(filename))
            data = nc.Dataset(os.path.basename(filename))
            if features[0] == 'sst':
                ncStLat = np.array(data.variables['lat'])[0]
                ncStLon = np.array(data.variables['lon'])[0]
                ncEdLat = np.array(data.variables['lat'])[-1]
                ncEdLon = np.array(data.variables['lon'])[-1]
            else:
                ncStLat = np.array(data.variables['latitude'])[0]
                ncStLon = np.array(data.variables['longitude'])[0]
                ncEdLat = np.array(data.variables['latitude'])[-1]
                ncEdLon = np.array(data.variables['longitude'])[-1]
            
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
                        regionaldata = data.variables[feature][:, :, int((startLat-ncStLat)/grid_size+0.1):int((endLat-ncStLat)/grid_size + 0.1)+1, int((startLon -ncStLon)/grid_size +0.1):int((endLon -ncStLon)/grid_size +0.1)+1]
                
                elif len(areaScope) == 2:
                    Lat = float(areaScope['lat'])
                    Lon = float(areaScope['lon'])
                    if Lon < 0:
                        Lon = Lon + 360
                    if feature=='sla':
                        regionaldata = data.variables[feature][:, int((Lat-ncStLat)/grid_size+0.1), int((Lon -ncStLon)/grid_size +0.1)]
                        regionaldata = regionaldata[0]
                    elif feature=='sst':
                        regionaldata = data.variables[feature][:, :, int((Lat-ncStLat)/grid_size+0.1), int((Lon -ncStLon)/grid_size +0.1)]
                        regionaldata = regionaldata[0][0]
                regionaldatas[feature] = regionaldata
            return regionaldatas
        
        except Exception as e:  
            print("执行 %s 函数发生错误：%s"  % (sys._getframe().f_code.co_name, e))
            return None
    
    #获取实测温度数据,并写入数据库 
    def read_wod_datas(self,filepath):
        depth,salt,temp,wod_datas = [],[],[],[]
        point = {}
        SSH_filename = os.path.join(self.Data_path,'SSH')
        id_count = 0
        psySQL.connectPostgreSQL()
        with open(filepath) as f:
            for line in f:
                line=line.strip()
                line_list = line.split()
                if len(line_list) >= 5:
                    #读取文件表头，格式（经度,纬度,层数,年,月,日,时（小时为浮点型））
                    code_dict = {}
                    code_dict['lon']=line_list[0]
                    code_dict['lat']=line_list[1]
                    code_dict['year']=line_list[3]
                    code_dict['month']=line_list[4]
                    code_dict['day']=line_list[5]
                    code_dict['datetime']=line_list[6]
                    date = arrow.get(code_dict['year']).shift(months=int(code_dict['month'])-1,days=int(code_dict['day'])-1).format(("YYYYMMDD"))
                    self.timeStatTnfo['endDay'] = date
                    print(str(date))
                    point['lat'] = code_dict['lat']
                    point['lon'] = code_dict['lon']
                    SSH = self.getSatelliteSSH(point)
                    SST = self.getSatelliteSST(point)
                    #SSH = self.read_satellite_datas(SSH_filename, ['sla'])
                    code_dict['SSH'] = SSH
                    code_dict['SST'] = SST
                    continue
                else:
                    code_dict['depth'] = line_list[0]
                    code_dict['temp'] = line_list[1]
                    code_dict['salt'] = line_list[2]
                    code_dict['ID'] = id_count
                    id_count += 1
                    sSQL = psySQL.selectOperate(id_count)
                    if sSQL == None:
                        psySQL.insertOperate(code_dict)
                    else:
                        continue
                print(id_count)      
        return id_count
    
def all_files_in_path(file_dir, filetype):   
    all_name=[]
    for _, _, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == filetype:  
                all_name.append(os.path.splitext(file)[0])
    return all_name


#timeStatInfo = {}
#timeStatInfo['timeStatType'] = 1
#timeStatInfo['statUnit'] = 2
#timeStatInfo['startDay'] = '20170101'
#timeStatInfo['endDay'] = '20171231'
#Data_path = 'D:/Datas'
#obj = load_data_and_labels(timeStatInfo, Data_path)
#wod = obj.read_wod_datas('D:\Datas\WOD\WOD13_2017.dat')
