# -*- coding: utf-8 -*-

#  获取日期名称列表
# July, 14, 2019
# 作者：tanghaoyu

import os
import sys
import io
import numpy as np 
import math

import matplotlib
from PIL.features import features
matplotlib.use('Agg') 
from matplotlib import pyplot as plt 

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
from PIL import Image
from cv2 import VideoWriter_fourcc
import cv2   

from common.config import CUR_PATH, _MAP_, _LAYOUT_, _DAT_, _PATH_, MAPSIZE, get_user_config, read_SYS_USER_option, get_CMAP, get_user_iso_interval
                          
from common.getDataset import NcData, is_invalid_array,split_features, getSectionMaxDepth
from common.dateFilesList import DaysList, getFileNameOfDay
from common.outFile import create_temp_filename
import common.ERR as err
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import AutoMinorLocator
import ssl

#title_font = {'family': 'SimHei', 'weight': 'bold', 'size': 15}  # 个性化定制某些字符的字体        标题用字体

# plt.rcParams['font.sans-serif'] = ['Deng']          #等线体（常规）
# plt.rcParams['axes.unicode_minus'] = False          #用来正常显示负号
plt.style.use(CUR_PATH + '/cora.mplstyle')

#要素名称获取轴线标注信息
def get_Axis_legend_label(feature):
    str_leg = _DAT_.FEATURE_CHI_NAME[feature] if feature in _DAT_.FEATURE_CHI_NAME else ''
    str_Unit = _DAT_.FEATURE_UNIT[feature] if feature in _DAT_.FEATURE_UNIT else ''
    
    axis_lab = str_leg
    if len(str_Unit)>0:
        axis_lab = str_leg + '(' + str_Unit + ')'
    
    return axis_lab, str_leg
    
#由要素名称获取标题名称
def get_title_label(feature):
    title = _DAT_.FEATURE_CHI_NAME[feature] if feature in _DAT_.FEATURE_CHI_NAME else ''
    return title 
    
# 基于数据集绘制大面图
# option：绘图参数化设置，包括投影、等高线等内容
# dataset： 输入数据集，格式： dataset:{codes:[,],values:[[,]]}
class RenderArea():
    def __init__(self, features, areaScope, datasets, grid_size, isoline = 0):                      #datasets内容多，features内容少
        self.features = []
        if features == '_ALL_':
            for key in datasets:
                self.features.append(key)
        else:
            if not isinstance(features, list):
                self.features.append(features)
            else:
                self.features = features
       
        self.minLat = float(areaScope['minLat'])
        self.maxLat = float(areaScope['maxLat'])
        self.minLon = float(areaScope['minLon'])
        self.maxLon = float(areaScope['maxLon'])   
                      
        if self.minLon > self.maxLon and self.maxLon < 0:                   #跨日经线，则转换经度范围为：[0~360}
            self.maxLon = 360 + self.maxLon
            
        self.datasets = datasets                                            # 要素名称为字典key值
        self.grid_size = grid_size
        self.projection = _MAP_.PROJECTION                                  # 读取系统定制投影名称        
        self.isoline = isoline
        self.interval = 0                                                   # 0表示未设置值
        self.title = ''
    
    #由要素名称、日期直接设置标题名称
    def setTitleOfMap(self, feature, day):
        scalarTitle = get_title_label(feature)
        if feature in ['UVEL','VVEL','flow']:
            self.title = '静态' + scalarTitle + '图('  + day + ')'
        else:
            self.title = scalarTitle + '大面图('  + day + ')'    
    
    #由要素名称确定专题图的标题         #有问题
    def getTitleOfMap(self):
        if self.title != '':                                                #title可以在外部设置
            return self.title
        
        scalarTitle, vectorTitle = '',''
        
        for feature in self.features:                                       #只取一个值，取到名称则退出
            scalarTitle = get_title_label(feature) + '大面图'
            if len(scalarTitle)>0:
                break
                               
        if 'UVEL' in self.features:
            vectorTitle = '流场'            
        
        if 'velocity' in self.features and 'UVEL' in self.features:
            title = vectorTitle
        elif len(scalarTitle)>0 and len(vectorTitle)>0:
            title = scalarTitle + '、' + vectorTitle
        elif len(scalarTitle)>0:
            title = scalarTitle
        elif len(vectorTitle)>0:
            title = vectorTitle
        else:
            title = '无'
        return title
        
    #确定地图绘制精度，暂定三个等级：l，i，h        
    def mapResolution(self, latWidth, lonWidth):
        width = min(latWidth, lonWidth)
        if width < 10:
            resolution = 'h'
        elif width < 50:
            resolution = 'i'
        else:
            resolution = 'l'
        return resolution
    
    #确定绘图尺寸，随着绘图区域的变化而变化
    def getwidth(self):
        if self.minLat == self.maxLat or self.minLon == self.maxLon:                #绘制区域范围
            return None
        
        latWidth = self.maxLat - self.minLat
        lonWidth = self.maxLon - self.minLon
                
        return latWidth, lonWidth
    
    #获取地图底图中心点
    def getBaseMapCenter(self):
        if (self.minLat+self.maxLat)==0 and _MAP_.PROJECTION=='lcc':                 #圆锥投影且区域为赤道两侧情况
            lat_0 = 0.01
        else:
            lat_0 = (self.minLat+self.maxLat)/2
        lon_0 = (self.minLon+self.maxLon)/2
        return lat_0, lon_0
    
    #过程中修改等值线参数
    def set_isoline(self, isoline, interval):
        key = [key for key in self.datasets if key not in ['UVEL','VVEL']]
        
        isoMax = np.max(self.datasets[key[0]])
        isoMin = np.min(self.datasets[key[0]])
        if (isoMax - isoMin)/interval > _DAT_.MAX_CONTOUR_LEVELS:               #15
            return False
        
        self.isoline = isoline
        self.interval = interval
        return True
    
    #过程中修改投影参数
    def set_projection(self, projection):
        self.projection = projection
        return True    
    
    def set_XY_ticks_lab(self, minPos, maxPos, nums):
        values = np.array([100,80,60,50,40,30,20,15,10,8,5,4,2,1,0.5,0.2,0.1,0])
        diff = maxPos - minPos
        interval = int(diff/nums)   #间隔值
        if interval < 0.1:          #间隔太小，不进行标注
            return []
        
        iPos = len(values[values > interval])
        interval = values[iPos]
        
        if minPos%interval >0:
            startPos = math.ceil(minPos/interval)*interval
            if (startPos - minPos)<interval/4:  #两者过于接近，离开一级
                startPos += interval
        else:
            startPos = minPos
            
        if maxPos%interval>0:
            endPos = (math.ceil(maxPos/interval)-1) * interval
            if (maxPos - endPos)<interval/4:
                endPos -= interval
        else:
            endPos = maxPos
        steps = int((endPos - startPos)/interval+0.1) + 1
        
        return np.linspace(startPos, endPos, steps)            
    
    # 利用定制底图，绘制大面图
    def create_basemap(self, resolution, user = 'guest', basemap = 'Common'):
        ssl._create_default_https_context = ssl._create_unverified_context
        
        if basemap == 'Pacific' and self.minLon >= 120 and self.maxLon <= 160 and self.minLat >= -2 and self.maxLat <= 46:
            server = 'https://thy.arcgis.cn:6443/arcgis'
            service = 'cora2/pacific'
            m = Basemap(llcrnrlon=self.minLon, urcrnrlon=self.maxLon, llcrnrlat=self.minLat, urcrnrlat=self.maxLat,resolution='h',epsg=4326)
            m.arcgisimage(server = server, service = service, xpixels=500, verbose=True)
        
        else:
            lat_0, lon_0 = self.getBaseMapCenter()                              #地图底图中心点坐标
            m = Basemap(llcrnrlon = self.minLon, urcrnrlon = self.maxLon, llcrnrlat = self.minLat, urcrnrlat = self.maxLat, lat_0 = lat_0,lon_0 = lon_0, projection=self.projection, resolution=resolution)
            m.drawmapboundary(fill_color = 'aqua')                            #无效值填充青绿色
            m.drawcoastlines()                                                #绘制海岸线
            #main_map.drawmapboundary()                                               #绘制图框
            m.fillcontinents(color='#CCCCCC')                                 #填充大陆  
        
        return m
    
    #主要函数，绘制大面图
    def RenderAreaDataset(self, user = 'guest', basemap = 'Common'): 
        try: 
            scaleFeature = ''  
            stream_plot = None
            b_color = False
            levels = ''
            
            latWidth, lonWidth = self.getwidth()
            read_SYS_USER_option('MAPSIZE', user)
            fig1 = plt.figure(figsize = (MAPSIZE.FIG_WIDTH, MAPSIZE.FIG_HEIGHT), dpi=MAPSIZE.DPI)                       #确定绘图大小、精度
            
            resolution = self.mapResolution(latWidth, lonWidth)                 #确定底图精度
            main_map = self.create_basemap(resolution, user, basemap)
                        
            lat_ticks_lab = self.set_XY_ticks_lab(self.minLat, self.maxLat, _LAYOUT_.AREA_LAB_Y)
            lon_ticks_lab = self.set_XY_ticks_lab(self.minLon, self.maxLon, _LAYOUT_.AREA_LAB_X)
            
            if len(lat_ticks_lab)>1:
                main_map.drawparallels(lat_ticks_lab, labels=[1,0,0,0], fontsize=10, fmt='%.0f', linewidth=1.)
            
            if len(lon_ticks_lab)>1:
                main_map.drawmeridians(lon_ticks_lab, labels=[0,0,0,1], fontsize=10, fmt='%.0f', linewidth=1.)   
            
            latSteps = int(latWidth/self.grid_size + 0.1) + 1
            lonSteps = int(lonWidth/self.grid_size + 0.1) + 1
            
            lat = list(np.linspace(self.minLat, self.maxLat, latSteps))
            lon = list(np.linspace(self.minLon, self.maxLon, lonSteps))
            
            x, y = np.meshgrid(lon,lat)         
            x, y = main_map(x,y)
            
            title = self.getTitleOfMap()
            plt.title(title)       #标题用黑体, fontdict = {'family':'SimHei'}
            
            if 'UVEL'  in self.features and 'VVEL' in self.features:
                uSpeed = self.datasets['UVEL']
                vSpeed = self.datasets['VVEL']
                stream_plot = main_map.streamplot(x, y, uSpeed, vSpeed,2, arrowsize=0.8, linewidth=0.8, color='blue')
                self.features.remove('UVEL')
                self.features.remove('VVEL')
            
            #渲染颜色      
            for feature in self.features:
                self.interval = get_user_iso_interval(feature, user)
                arr, labLevles, isoMax, isoMin = get_iso_labels(self.datasets[feature], feature, self.interval)
                
                cmap = get_CMAP(feature, 'MAP', user)
                if cmap['Type'] == 'discrete':                                          #非连续，表示为阶梯色
                    if isinstance(cmap['value'], list):                                 #如果读出的色标是颜色值列表
                        levels = cmap['Levels']
                        cs = main_map.contourf(x,y,self.datasets[feature],levels = levels, colors=cmap['value'])
                    else:
                        cs = main_map.contourf(x,y,self.datasets[feature],vmin = isoMin, vmax = isoMax,  cmap=cmap['value'])
                
                else:
                    cs = main_map.pcolormesh(x,y,self.datasets[feature],vmin = isoMin, vmax = isoMax,  cmap=cmap['value'])
                
                scaleFeature = feature
                break
                    
            # 绘制等值线
            if stream_plot == None and self.isoline == 1 and len(scaleFeature) > 0:     #绘制静态流场图后就不能绘等值线图
                
                ct = plt.contour(x, y, arr, labLevles,linewidths = _MAP_.LINEWIDTH, vmin = isoMin, vmax = isoMax, colors = _MAP_.LINECOLOR)
                plt.clabel(ct, inline = _MAP_.INLINE, fontsize = _MAP_.FONTSIZE, fmt = '%0.1f') 
            
            #绘制色标
            if len(scaleFeature) > 0:
                divider = make_axes_locatable(plt.gca())                            #获取当前视图轴线
                cax = divider.append_axes("right", 0.3, pad=0.2)                    #色标轴的位置：left|right|bottom|top；size = 0.3,轴宽度；pad:标注间隔          
                cb=plt.colorbar(cs, shrink=0.85, cax=cax)                           #色标  
                #cb=main_map.colorbar(cs)   #, shrink=0.85,, cax=cax                        #色标      
                cb.ax.tick_params(labelsize=8)                                      #设置色标刻度字体大小
#                 font = {'family' : 'serif', 
#                         'color' : 'darkred', 
#                         'weight' : 'normal', 
#                         'size' : 8, } 
                if len(levels)>0:
                    ticklabels = [str(level) for level in levels if level < 9999 and level > -9999]
                    ticklabels = ['<' + ticklabels[0]] + ticklabels + ['>' + ticklabels[-1]]
                    
                    cb.set_ticklabels(ticklabels)                                 #设置colorbar的标签字体及其大小   ,fontdict=font
            
            rect = [0.04, 0.02, 1, 1] 
            fig1.tight_layout(rect=rect)   #w_pad = 0.1, h_pad= 0.1                 #调整子图，以使与主图相配适合，左边、下边留出一定空白
            
            canvas = fig1.canvas
            buffer = io.BytesIO()
            canvas.print_png(buffer)
            data=buffer.getvalue()
            
            ###########################################################################################
            img=Image.open(buffer)
            img = np.asarray(img)
            new_img = img[:,:,0:3]
 
            '''判断图像边距'''
            top_padding,bottom_padding,left_padding,right_padding = Frame_detecting(new_img, "Area")
            padding = {'width': MAPSIZE.FIG_WIDTH * MAPSIZE.DPI, 'height': MAPSIZE.FIG_HEIGHT * MAPSIZE.DPI ,
                       'top':top_padding, 'bottom':bottom_padding, 'left':left_padding, 'right': right_padding}
            ###########################################################################################
            
            buffer.close()
            plt.close()
            
            return data, padding
        except Exception as e:  
            print("执行 %s 函数发生错误：%s"  % (sys._getframe().f_code.co_name, e))
            return False  
    
    #主要函数，绘制漂流浮标流线图
    def create_drifters_map(self, user = 'guest'): 
        try:   
            latWidth, lonWidth = self.getwidth()
            read_SYS_USER_option('MAPSIZE', user)
            fig1 = plt.figure(figsize = (MAPSIZE.FIG_WIDTH, MAPSIZE.FIG_HEIGHT), dpi=MAPSIZE.DPI)                       #确定绘图大小、精度
            
            resolution = self.mapResolution(latWidth, lonWidth)                 #确定底图精度
            lat_0, lon_0 = self.getBaseMapCenter()                              #地图底图中心点坐标
            main_map = Basemap(llcrnrlon = self.minLon, urcrnrlon = self.maxLon, llcrnrlat = self.minLat, urcrnrlat = self.maxLat, lat_0 = lat_0,lon_0 = lon_0, projection=self.projection, resolution=resolution)
            main_map.drawcoastlines()                                           #绘制图框
            main_map.fillcontinents(color='#CCCCCC')                                 #填充大陆  
            main_map.drawmapboundary(fill_color = 'aqua')                            #无效值填充青绿色
            
            lat_ticks_lab = self.set_XY_ticks_lab(self.minLat, self.maxLat, _LAYOUT_.AREA_LAB_Y)
            lon_ticks_lab = self.set_XY_ticks_lab(self.minLon, self.maxLon, _LAYOUT_.AREA_LAB_X)
            
            if len(lat_ticks_lab)>1:
                main_map.drawparallels(lat_ticks_lab, labels=[1,0,0,0], fontsize=10, fmt='%.0f', linewidth=1.)
            
            if len(lon_ticks_lab)>1:
                main_map.drawmeridians(lon_ticks_lab, labels=[0,0,0,1], fontsize=10, fmt='%.0f', linewidth=1.)   
            
            x, y = main_map(self.datasets['obs_lons'], self.datasets['obs_lats'])
            plt.plot(x, y, color='b', label='观测值') 
            
            m, n = main_map(self.datasets['cal_lons'], self.datasets['cal_lats'])
            plt.plot(m, n, color='r', label='计算值') 
                                  
            x, y = main_map(self.datasets['obs_lons'][0], self.datasets['obs_lats'][0])
            main_map.plot(x, y, marker='D',color='m')
            
            plt.title('漂流轨迹对比图')
            
            #图例绘制
            plt.legend(loc='upper right', shadow=True, fontsize = 9)
            
            rect = [0.04, 0.02, 1, 1] 
            fig1.tight_layout(rect=rect) 
            
            canvas = fig1.canvas
            buffer = io.BytesIO()
            canvas.print_png(buffer)
            data=buffer.getvalue()
            
            ###########################################################################################
            img=Image.open(buffer)
            img = np.asarray(img)
            new_img = img[:,:,0:3]
 
            '''判断图像边距'''
            top_padding,bottom_padding,left_padding,right_padding = Frame_detecting(new_img, "Area")
            padding = {'width': MAPSIZE.FIG_WIDTH * MAPSIZE.DPI, 'height': MAPSIZE.FIG_HEIGHT * MAPSIZE.DPI ,
                       'top':top_padding, 'bottom':bottom_padding, 'left':left_padding, 'right': right_padding}
            ###########################################################################################
            
            buffer.close()
            plt.close()
            
            return data, padding
        except Exception as e:  
            print("执行 %s 函数发生错误：%s"  % (sys._getframe().f_code.co_name, e))
            return False  
            
                  

#绘制基于地图的散点图
class RenderScatter():
    def __init__(self, areaScope, points, values):
        lats, lons = [], []
        self.minLat = float(areaScope['minLat'])
        self.maxLat = float(areaScope['maxLat'])
        self.minLon = float(areaScope['minLon'])
        self.maxLon = float(areaScope['maxLon'])
        
        for point in points:
            lat = point['lat']
            lon = point['lon']
            lats.append(lat)
            lons.append(lon)
        self.lats = lats 
        self.lons = lons
        self.values = values
        #确定地图绘制精度，暂定三个等级：l，i，h        
    
    def mapResolution(self, latWidth, lonWidth):
        width = min(latWidth, lonWidth)
        if width < 10:
            resolution = 'h'
        elif width < 50:
            resolution = 'i'
        else:
            resolution = 'l'
        return resolution
    
    #确定绘图尺寸，随着绘图区域的变化而变化
    def getwidth(self):
        if self.minLat == self.maxLat or self.minLon == self.maxLon:                #绘制区域范围
            return None
        
        latWidth = self.maxLat - self.minLat
        lonWidth = self.maxLon - self.minLon
                
        return latWidth, lonWidth
    
    #获取地图底图中心点
    def getBaseMapCenter(self):
        if (self.minLat+self.maxLat)==0 and _MAP_.PROJECTION=='lcc':                 #圆锥投影且区域为赤道两侧情况
            lat_0 = 0.01
        else:
            lat_0 = (self.minLat+self.maxLat)/2
        lon_0 = (self.minLon+self.maxLon)/2
        return lat_0, lon_0
    
    def set_XY_ticks_lab(self, minPos, maxPos, nums):
        values = np.array([100,80,60,50,40,30,20,15,10,8,5,4,2,1,0.5,0.2,0.1,0])
        diff = maxPos - minPos
        interval = int(diff/nums)   #间隔值
        if interval < 0.1:          #间隔太小，不进行标注
            return []
        
        iPos = len(values[values > interval])
        interval = values[iPos]
        
        if minPos%interval >0:
            startPos = math.ceil(minPos/interval)*interval
            if (startPos - minPos)<interval/4:  #两者过于接近，离开一级
                startPos += interval
        else:
            startPos = minPos
            
        if maxPos%interval>0:
            endPos = (math.ceil(maxPos/interval)-1) * interval
            if (maxPos - endPos)<interval/4:
                endPos -= interval
        else:
            endPos = maxPos
        steps = int((endPos - startPos)/interval+0.1) + 1
        
        return np.linspace(startPos, endPos, steps)
    
#     #按照正负分为两部分，以赋予不同颜色
#     def split_two_parts(self):
#         p_values, n_values = [],[]
#         p_lats, p_lons, n_lats, n_lons = [],[],[],[]
#         p_scatter, n_scatter={},{}
#         
#         for i, value in enumerate(self.values):
#             if value >= 0:
#                 p_values.append(value)
#                 p_lats.append(self.lats[i])
#                 p_lons.append(self.lons[i])
#             else:
#                 n_values.append(-value)                                         #负数转正值，但归入负数组
#                 n_lats.append(self.lats[i])
#                 n_lons.append(self.lons[i])
#         p_scatter['values'] = np.array(p_values)
#         p_scatter['lats'] = np.array(p_lats)
#         p_scatter['lons'] = np.array(p_lons)
#         
#         n_scatter['values'] = np.array(n_values)
#         n_scatter['lats'] = np.array(n_lats)
#         n_scatter['lons'] = np.array(n_lons)
#         
#         return p_scatter, n_scatter
    
    #参数：values，为矩阵
    #min对应minsize, max对应maxsize，归一化处理
    def normalize(self, values, minsize, maxsize):
        iLen = len(values)
        min_v = np.min(values)
        max_v = np.max(values)

        arr_min = np.ones(iLen) * min_v
        arr_minsize = np.ones(iLen) * minsize
        
        new_values = (values- arr_min) * ((maxsize -minsize)/(max_v - min_v)) + arr_minsize
        return new_values
        
    #绘制散点地图
    def create_scatter_map(self, user = 'guest', map_type = 'hexbin'):
        try: 
            latWidth, lonWidth = self.getwidth()
            read_SYS_USER_option('MAPSIZE', user)
            fig1 = plt.figure(figsize = (MAPSIZE.FIG_WIDTH, MAPSIZE.FIG_HEIGHT), dpi=MAPSIZE.DPI)                       #确定绘图大小、精度
            
            resolution = self.mapResolution(latWidth, lonWidth)                 #确定底图精度
            lat_0, lon_0 = self.getBaseMapCenter()                              #地图底图中心点坐标
            main_map = Basemap(llcrnrlon = self.minLon, urcrnrlon = self.maxLon, llcrnrlat = self.minLat, urcrnrlat = self.maxLat, lat_0 = lat_0,lon_0 = lon_0, projection=_MAP_.PROJECTION, resolution=resolution)
            main_map.drawcoastlines()                                           #绘制海岸线
            main_map.fillcontinents(color='#CCCCCC')                            #填充大陆  
            main_map.drawmapboundary(fill_color = 'aqua')                                          
            
            lat_ticks_lab = self.set_XY_ticks_lab(self.minLat, self.maxLat, _LAYOUT_.AREA_LAB_Y)
            lon_ticks_lab = self.set_XY_ticks_lab(self.minLon, self.maxLon, _LAYOUT_.AREA_LAB_X)
            
            if len(lat_ticks_lab)>1:
                main_map.drawparallels(lat_ticks_lab, labels=[1,0,0,0], fontsize=10, fmt='%.0f', linewidth=1.)
            
            if len(lon_ticks_lab)>1:
                main_map.drawmeridians(lon_ticks_lab, labels=[0,0,0,1], fontsize=10, fmt='%.0f', linewidth=1.) 
                
            x,y = main_map(self.lons, self.lats)
            
            if map_type == 'hexbin':            
                cs = main_map.hexbin(np.array(x), np.array(y), C = np.array(self.values), gridsize=30, cmap='inferno', mincnt =1, alpha = 0.6)    #cmpa = 'summer'
                
                #绘制色标
                divider = make_axes_locatable(plt.gca())                            #获取当前视图轴线
                cax = divider.append_axes("right", 0.3, pad=0.2)                    #色标轴的位置：left|right|bottom|top；size = 0.3,轴宽度；pad:标注间隔          
                cb=plt.colorbar(cs, shrink=0.85, cax=cax)                           #色标        
                cb.ax.tick_params(labelsize=8)                                      #设置色标刻度字体大小
                font = {'family' : 'serif', 
                        'color' : 'darkred', 
                        'weight' : 'normal', 
                        'size' : 8, } 
                cb.set_label('',fontdict=font)                                      #设置colorbar的标签字体及其大小
            
            else:       
                size = self.normalize(self.values, 10, 100)
                main_map.scatter(x,y,size, color='b', alpha = 0.5)
            
            plt.title('区域误差散点图')
            
            rect = [0.04, 0.02, 1, 1] 
            fig1.tight_layout(rect=rect)   #w_pad = 0.1, h_pad= 0.1                 #调整子图，以使与主图相配适合，左边、下边留出一定空白
            
            canvas = fig1.canvas
            buffer = io.BytesIO()
            canvas.print_png(buffer)
            data=buffer.getvalue()
            
            ###########################################################################################
            img=Image.open(buffer)
            img = np.asarray(img)
            new_img = img[:,:,0:3]
 
            '''判断图像边距'''
            top_padding,bottom_padding,left_padding,right_padding = Frame_detecting(new_img, "Area")
            padding = {'width': MAPSIZE.FIG_WIDTH * MAPSIZE.DPI, 'height': MAPSIZE.FIG_HEIGHT * MAPSIZE.DPI ,
                       'top':top_padding, 'bottom':bottom_padding, 'left':left_padding, 'right': right_padding}
            ###########################################################################################
            
            buffer.close()
            plt.close()
            
            return data, padding
        except Exception as e:  
            print("执行 %s 函数发生错误：%s"  % (sys._getframe().f_code.co_name, e))
            return False,False        

#断面渲染图，目前只能处理一个要素信息，处理断面流场图有问题，2至3个要素：流速、UVEL/VVEL、WVEL                   
class RenderSection():
    def __init__(self, feature, SectionLine, depthList, datasets, grid_size, isoline = 0):
        self.feature = feature
        self.datasets = datasets
        
        self.leftPtLat  = float(SectionLine['leftPtLat'])
        self.leftPtLon  = float(SectionLine['leftPtLon'])
        self.rightPtLat = float(SectionLine['rightPtLat'])
        self.rightPtLon = float(SectionLine['rightPtLon']) 
        
        if self.leftPtLon == self.rightPtLon:
            self.X_Axis = 'LAT'                                                 #纯纬向断面
            if self.leftPtLat > self.rightPtLat:                                #确保纯经向断面的左点在下，右点在上
                temp = self.rightPtLat
                self.rightPtLat = self.leftPtLat
                self.leftPtLat = temp
        elif self.leftPtLat == self.rightPtLat:
            self.X_Axis = 'LON'                                                 #纯经向断面
        else:
            self.X_Axis = 'LONLAT'                                              #以LON值定位，但经纬度值均标注
            
        if self.leftPtLon > self.rightPtLon and self.rightPtLon < 0:            #一次性解决跨日经线问题
            self.rightPtLon = 360 + self.rightPtLon
            
        self.depthList = depthList
        self.grid_size = grid_size
                                                         
        #read_SYS_USER_option(['_LAYOUT_','_MAP_'])
        self.isoline = isoline
        self.interval = 0
        self.title = ''
        
    #由要素名称、日期直接设置标题名称
    def setTitleOfMap(self, feature, day):
        sTitle = get_title_label(feature)
        self.title = sTitle + '断面图('  + day + ')'
        
    # 获取断面图的网格坐标数据
    def getAxis(self):
        lat_steps = int(abs(self.rightPtLat - self.leftPtLat)/self.grid_size +0.1) + 1
        lon_steps = int((self.rightPtLon - self.leftPtLon)/self.grid_size +0.1) + 1
                
        if lon_steps == 1:
            axis_x = np.linspace(self.leftPtLat, self.rightPtLat, lat_steps)
        else:
            x_steps = max(lat_steps, lon_steps)    
            axis_x = np.linspace(self.leftPtLon, self.rightPtLon, x_steps)
        
        axis_y = -np.array(self.depthList)                                      #先转为负数，以便 0 值 在最上方
        return axis_x, axis_y
    
    #根据横纵轴间隔数量确定两轴标注的内容
    def getXYLabel(self, xNum, yNum):
        try:
            xList = []                                                              # 横纵轴标注文字列表
            if self.X_Axis == 'LON':
                lonText, xNum = self.set_XY_ticks_lab(self.leftPtLon, self.rightPtLon, xNum)            
            
            elif self.X_Axis == 'LAT':
                latText, xNum = self.set_XY_ticks_lab(self.leftPtLat, self.rightPtLat, xNum)            
            
            else:
                latText = np.linspace(self.leftPtLat, self.rightPtLat, xNum)
                lonText = np.linspace(self.leftPtLon, self.rightPtLon, xNum)
                
            for i in range(xNum):
                if self.X_Axis == 'LAT':            #只标注纬度值
                    if latText[i]>0:
                        latUnit='°N'
                    elif latText[i]<0:
                        latUnit='°S'
                    else:
                        latUnit='°'
                    latLabel = str(round(abs(latText[i]),1)) + latUnit
                    xList.append(latLabel)
                    
                elif self.X_Axis == 'LON':          #只标注经度值
                    if lonText[i]<180 and lonText[i]>=0:
                        lon_text = lonText[i]
                        lonUnit='°E'
                    elif lonText[i]>=180:
                        lon_text = 360. - lonText[i]
                        lonUnit='°W'
                    else:                           #负数
                        lon_text = -lonText[i]
                        lonUnit='°W'
                        
                    lonLabel = str(round(abs(lon_text),1)) + lonUnit
                    xList.append(lonLabel)
                    
                else:                               #纬经数据均标注            
                    if latText[i]>0:
                        latUnit='°N'
                    elif latText[i]<0:
                        latUnit='°S'
                    else:
                        latUnit='°'
                        
                    if lonText[i]<180 and lonText[i]>=0:
                        lon_text = lonText[i]
                        lonUnit='°E'
                    elif lonText[i]>=180:
                        lon_text = 360. - lonText[i]
                        lonUnit='°W'
                    else:                           #负数
                        lon_text = -lonText[i]
                        lonUnit='°W'
                         
                    latLabel = str(round(abs(latText[i]),1)) + latUnit
                    lonLabel = str(round(abs(lon_text),1)) + lonUnit            #lonText显示值有修改，采用中间变量处理
                    
                    xList.append(latLabel + '\n' + lonLabel)
            
            minDepthValue = int(abs(self.depthList[0]) + 0.5)
            maxDepthValue = int(abs(self.depthList[-1]) + 0.5)
            
            yList = self.set_depth_lab(minDepthValue, maxDepthValue, yNum) 
            yList = yList.astype(np.int)
            
            #设置标注位置
            if self.X_Axis =='LAT':
                xPos = latText
            else:
                xPos = lonText
            
            yPos = -yList
                
            return xList, yList, xPos, yPos   
        except Exception as e:  
            print("执行 %s 函数发生错误：%s"  % (sys._getframe().f_code.co_name, e))
            return [],[],[],[]
                
    #设置经纬度数据，仅经向、纬向断面图时用
    def set_XY_ticks_lab(self, firstPos, secondPos, nums):
        values = np.array([100,80,60,50,40,30,20,15,10,5,2,1,0.5,0.2,0.1,0])
        diff = abs(secondPos - firstPos)
        interval = int(diff/nums)   #间隔值
        if interval < 0.1:          #间隔太小，不进行标注
            raise Exception("差值太小，无法生成数据列表")
            return [],None
        
        iPos = len(values[values > interval])
        interval = values[iPos]
        
        if secondPos <firstPos:
            interval = -interval
        
        if firstPos%abs(interval) >0:
            startPos = math.ceil(firstPos/interval) * interval
        else:
            startPos = firstPos
                
        if secondPos%abs(interval)>0:
            endPos = (math.ceil(secondPos/interval)-1) * interval
        else:
            endPos = secondPos
            
        steps = int((endPos - startPos)/interval+0.1) + 1
        
        return np.linspace(startPos, endPos, steps), steps
    
    #深度值列表标准化处理
    def set_depth_lab(self, mindepth, maxdepth, nums):
        values = np.array([2000,1500,1200,1000,800,600,500,400,300,200,100,80,60,50,40,30,20,15,10,5])
        diff = maxdepth - mindepth
        interval = int(diff/nums)   #间隔值
        if interval < 5:          #间隔太小，不进行标注
            return []
        
        iPos = len(values[values > interval])
        interval = values[iPos]
        
        if mindepth%interval >0:
            startPos = math.ceil(mindepth/interval) * interval
        else:
            startPos = mindepth

        if maxdepth%interval>0:
            endPos = (math.ceil(maxdepth/interval)-1) * interval
        else:
            endPos = maxdepth
            
        steps = int((endPos - startPos)/interval+0.1) + 1
        
        yList = np.linspace(startPos, endPos, steps).astype(np.int)
        return yList
           
    # 根据标注数量确定轴线标注数值位置列表            
#     def getLabelPos(self, axis_x , xNum):        
#         minPos = axis_x[0]
#         maxPos = axis_x[-1]
#         labelPos = np.linspace(minPos, maxPos, xNum)
#         return labelPos

    #绘制断面流场图（未处理绘制静态箭头流场）
    def render_vector_section(self, user = 'guest'):
        try:
            axis_x, axis_y = self.getAxis()                                                #获取X、Y轴坐标值，Y值改为负数，以确保表面数据在上方
            x,y = np.meshgrid(axis_x,axis_y)
            
            xList, yList, xLabelPos, yLabelPos= self.getXYLabel(_LAYOUT_.SEC_LAB_X, _LAYOUT_.SEC_LAB_Y)        
            #xList, yList = self.getXYLabel(_LAYOUT_.SEC_LAB_X, _LAYOUT_.SEC_LAB_Y)  
            #xLabelPos = self.getLabelPos(axis_x,  _LAYOUT_.SEC_LAB_X)                               #获取横轴标注位置列表
            #yLabelPos = self.getLabelPos(axis_y,  _LAYOUT_.SEC_LAB_Y)                               #获取纵轴标注位置列表
            
            #########################################################################################################################
            read_SYS_USER_option('MAPSIZE', user)
            fig1 = plt.figure(figsize=[MAPSIZE.FIG_WIDTH, MAPSIZE.FIG_HEIGHT], dpi=MAPSIZE.DPI)
            plt.subplot(111,facecolor ='#BFEFFF')                                           #无效值设置颜色  
            
            #title = get_title_label('velocity')
            plt.title('断面流速图') 

            if self.X_Axis == 'LON':
                plt.xlabel('经度(°E)')
            elif self.X_Axis == 'LAT': 
                plt.xlabel('纬度(°N)')
            else:
                plt.xlabel('纬经度(°N,°E)')                                                    #中文标注有问题
            plt.ylabel('水深(米)')
            
            plt.xticks(xLabelPos, xList)                                                    #改变横轴标注内容        
            plt.yticks(yLabelPos, yList)
            
            #渲染流速大面与箭头流线图
            uSpeed = self.datasets['XVEL']
            wSpeed = self.datasets['WVEL']
            
            velocity = np.sqrt(uSpeed*uSpeed + wSpeed*wSpeed)
            
            levels = MaxNLocator(nbins=40).tick_values(velocity.min(), velocity.max())
            
            cmap = get_CMAP('velocity', 'MAP', user)
            cs = plt.contourf(x, y, velocity, levels=levels, cmap = cmap['value'])
    
            if self.isoline == 1:
                if self.interval == 0:
                    self.interval = get_user_config('ISOLINE_INTV', 'velocity', user)
                    
                arr, labLevles, isoMax, isoMin =  get_iso_labels(velocity, 'velocity', self.interval)          #获取等值线标注信息
                 
                ct = plt.contour(x, y, arr, labLevles, linewidths = _MAP_.LINEWIDTH, vmin = isoMin, vmax = isoMax, colors=_MAP_.LINECOLOR)
                plt.clabel(ct, inline = _MAP_.INLINE, fontsize=_MAP_.FONTSIZE, colors= _MAP_.FONTCOLOR) #, alpha = mapOption['fontAlpha']   
                
            #绘制色标
            divider = make_axes_locatable(plt.gca())                            #获取当前视图轴线
            cax = divider.append_axes("right", 0.3, pad=0.2)                    #色标轴的位置：left|right|bottom|top；size = 0.3,轴宽度；pad:标注间隔          
            cb = plt.colorbar(cs, shrink=0.85, cax=cax)                         #色标        
            cb.ax.tick_params(labelsize=8)                                      #设置色标刻度字体大小
            font = {'family' : 'serif', 
                    'color' : 'darkred', 
                    'weight' : 'normal', 
                    'size' : 8, } 
            cb.set_label('',fontdict=font)                                      #设置colorbar的标签字体及其大小
                              
            plt.tight_layout()                                                  #调整子图，以使与主图相配适合
            canvas = fig1.canvas
            buffer = io.BytesIO()
            canvas.print_png(buffer)
            data=buffer.getvalue()
            
            img=Image.open(buffer)
            img = np.asarray(img)
            new_img = img[:,:,0:3]
            
            '''判断图像边距'''
            top_padding,bottom_padding,left_padding,right_padding = Frame_detecting(new_img, "Area")
            padding = {'width': MAPSIZE.FIG_WIDTH * MAPSIZE.DPI, 'height': MAPSIZE.FIG_HEIGHT * MAPSIZE.DPI ,
                   'top':top_padding, 'bottom':bottom_padding, 'left':left_padding, 'right': right_padding}
            
            buffer.close()
            
            return data, padding
        
        except Exception as e:  
            print("执行 %s 函数发生错误：%s"  % (sys._getframe().f_code.co_name, e))
            return None
    
    #过程中修改等值线参数
    def set_isoline(self, isoline, interval):
        self.isoline = isoline
        self.interval = interval
        return True
               
    #绘制断面图
    def RenderSectionDataset(self, user='guest'):
        try:
            #只有一个要素，所以判断处理一个self.datasets，针对重绘断面图有意义
            if len(self.datasets) == 1:
                for key in self.datasets:
                    self.datasets = self.datasets[key]
            
            axis_x, axis_y = self.getAxis()                                                         #获取X、Y轴坐标值，Y值改为负数，以确保表面数据在上方
            x,y = np.meshgrid(axis_x,axis_y)
                    
            xList, yList, xLabelPos, yLabelPos= self.getXYLabel(_LAYOUT_.SEC_LAB_X, _LAYOUT_.SEC_LAB_Y)
            
            #########################################################################################################################
            read_SYS_USER_option('MAPSIZE', user)
            fig1 = plt.figure(figsize=[MAPSIZE.FIG_WIDTH, MAPSIZE.FIG_HEIGHT], dpi=MAPSIZE.DPI)     #绘图宽度为7inch，是否合适？7、80应作为全局变量
            plt.subplot(111,facecolor ='#BFEFFF')                                                   #无效值设置颜色
            
            if self.title == '':
                s_feature = _DAT_.FEATURE_CHI_NAME[self.feature] if self.feature in _DAT_.FEATURE_CHI_NAME else self.feature
                self.title = '断面图（%s）'  % s_feature                                             #get_title_label(self.feature)
            plt.title(self.title)                                                                   #中文标注有问题
            plt.xticks(xLabelPos, xList)                                                            #改变横轴标注内容        
            plt.yticks(yLabelPos, yList)
            
            if self.X_Axis == 'LON':
                plt.xlabel('经度(°E)')
            elif self.X_Axis == 'LAT': 
                plt.xlabel('纬度(°N)')
            else:
                plt.xlabel('纬经度(°N,°E)')
            plt.ylabel('深度(米)')
            
            #渲染颜色
            levels = MaxNLocator(nbins=40).tick_values(self.datasets.min(), self.datasets.max())
            
            cmap= get_CMAP(self.feature, 'MAP', user)
#             if cmap == False:
#                 cmap = get_CMAP('OTHER', 'MAP', user)                                               #没取到色标，则取OTHER定义的色标值 
                        
            cs = plt.contourf(x, y, self.datasets, levels=levels, cmap = cmap['value'])            
            
            # 绘制等值线
            if self.isoline == 1:
                arr, labLevles, isoMax, isoMin =  get_iso_labels(self.datasets, self.feature, self.interval)          #获取等值线标注信息
                
                ct = plt.contour(x, y, arr, labLevles, linewidths = _MAP_.LINEWIDTH, vmin = isoMin, vmax = isoMax, colors=_MAP_.LINECOLOR)
                plt.clabel(ct, inline = _MAP_.INLINE, fontsize=_MAP_.FONTSIZE, colors= _MAP_.FONTCOLOR, fmt= '%1.1f')   
                
            #绘制色标
            divider = make_axes_locatable(plt.gca())                                                #获取当前视图轴线
            cax = divider.append_axes("right", 0.3, pad=0.2)                                        #色标轴的位置：left|right|bottom|top；size = 0.3,轴宽度；pad:标注间隔          
            cb = plt.colorbar(cs, shrink=0.85, cax=cax)                                             #色标        
            cb.ax.tick_params(labelsize=8)                                                          #设置色标刻度字体大小
            font = {'family' : 'serif', 
                    'color' : 'darkred', 
                    'weight' : 'normal', 
                    'size' : 8, } 
            cb.set_label('',fontdict=font)                                      #设置colorbar的标签字体及其大小
                            
            plt.tight_layout()                                                  #调整子图，以使与主图相配适合
            canvas = fig1.canvas #. plt.get_current_fig_manager().canvas        #写入内存
    #        canvas.draw()
            buffer = io.BytesIO()
            canvas.print_png(buffer)
            data=buffer.getvalue()
            
            img=Image.open(buffer)
            img = np.asarray(img)
            new_img = img[:,:,0:3]
            
            '''判断图像边距'''
            top_padding,bottom_padding,left_padding,right_padding = Frame_detecting(new_img, "Area")
            padding = {'width':MAPSIZE.FIG_WIDTH*MAPSIZE.DPI, 'height':MAPSIZE.FIG_HEIGHT*MAPSIZE.DPI ,
                   'top':top_padding, 'bottom':bottom_padding, 'left':left_padding, 'right': right_padding}
            
            buffer.close()
            
            return data, padding
        except Exception as e:  
            print("执行 %s 函数发生错误：%s"  % (sys._getframe().f_code.co_name, e))
            return None,None

#获取等值线标注文字列表信息
#改造：由绘制频率改为间隔值
def get_iso_labels(dataset, feature, interval):        
    #等值线最大值，应该取读数的最大值，取最近整数或小数（根据feature而定)
    isoMax = np.nanmax(dataset)
    isoMin = np.nanmin(dataset)
    if interval == 0:
        return dataset, [], isoMax, isoMin
    
    isoMin = np.ceil(isoMin/interval)*interval
    isoMax = int((isoMax/interval)*interval)
#     
#     if feature in ['ETAN', 'SALTanom', 'THETA']:
#         isoMax = int(isoMax + 0.5)
#         isoMin = int(isoMin + 0.5)
#         
#     elif feature == 'velocity':
#         isoMax = int(isoMax*10)/10
#         isoMin = int(isoMin*10)/10
#         
#     elif feature == 'rho':
#         isoMax = int((isoMax -1000)/10 + 0.5)*10 + 1000
#         isoMin = int((isoMin -1000)/10 + 0.5)*10 + 1000
#     elif feature == 'sspd':
#         isoMax = int((isoMax -1400)/10 + 0.5)*10 + 1400
#         isoMin = int((isoMin -1400)/10 + 0.5)*10 + 1400
#         
    #isoStep =  _MAP_.ISOSTEP                               #根据绘制频率确定等值距
     
    arr = dataset
#     arr[arr>isoMax] = float(isoMax)
#     arr[arr<isoMin] = float(isoMin)                         #绘制等值线，
    
    steps = int((isoMax - isoMin)/interval+0.1)+1
    
    #labLevles = np.arange(isoMin, isoMax, interval)        #确定等值标注列表
    labLevles = np.linspace(isoMin, isoMax, steps)          #确定等值标注列表
    return arr, labLevles, isoMax, isoMin

#创建标量场图（大面、断面）Mp4、AVI
#属性：文件类型、产品类型、时间范围、区域范围、深度、要素
#fileType = '.avi' / '.mp4
class RenderVideoScalar():
    def __init__(self, fileType, prodType, timeScope, areaScope, depth, features): 
        self.fileType = fileType
        self.prodType = prodType
        self.grid_size = _DAT_.CA_GRID_SIZE if prodType =='CA' else _DAT_.RA_GRID_SIZE
        self.timeScope = timeScope
        self.areaScope = areaScope
        self.depth = depth
        self.features = features
        self.title = ''
        #read_SYS_USER_option(['_LAYOUT_','_MAP_'])
        
    def get_day_list(self):
        startDay = self.timeScope['startDay']
        endDay = self.timeScope['endDay']
        minYear = int(startDay[:4])
        maxYear = int(endDay[:4])
        minMonth = int(startDay[4:6])
        maxMonth = int(endDay[4:6])
        minDay = int(startDay[-2:])
        maxDay = int(endDay[-2:])
        
        obj = DaysList(1, minYear, maxYear, minMonth, maxMonth, minDay, maxDay)     #参数1，表示获取时间范围
        all_days = obj.AllDaysList()
        return all_days
    
    #获取存放动态文件的目录，如不存在则创建一个
    def get_temp_path(self):
        dyn_path = _PATH_.USER
        if not os.path.exists(dyn_path):
            os.makedirs(dyn_path)
        return dyn_path
    
    def create_video_writer(self, user):
        read_SYS_USER_option('MAPSIZE', user)
        video_size = (int(MAPSIZE.FIG_WIDTH * MAPSIZE.DPI), MAPSIZE.FIG_HEIGHT * MAPSIZE.DPI)
        fps = _LAYOUT_.VIDEO_FPS                                                                  #每秒帧数 
        if self.fileType == '.avi':
            fourcc = VideoWriter_fourcc(*"MJPG")                                #定义编码器
        elif self.fileType == '.mp4':
            #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            fourcc = cv2.VideoWriter_fourcc('A', 'V', 'C', '1')
                    
        video_Name = create_temp_filename()                                        #以时间戳命名，用~标识临时文件
        temp_Path = self.get_temp_path()
        video_name = os.path.join(temp_Path, user, video_Name + self.fileType)
        videoWriter = cv2.VideoWriter(video_name, fourcc, fps, video_size)        #参数：保存文件名、编码器、帧率、视频宽高
        return video_name, videoWriter    
            
    #生成标量大面图Dynamic文件
    def create_video_scalar_field(self, user):
        try:   
            b_flow = False
            
            pic_seconds =  _DAT_.PIC_SECONDS #get_user_config('MISC', 'PIC_SECONDS', user)            #每个画面停留时长
            
            err_count = 0
            datasetType = 'DatasetofHArea'
            video_name, videoWriter = self.create_video_writer(user)
            
            all_days = self.get_day_list()
            if self.features == ['flow']:
                b_flow = True
                
            for day in all_days:
                frames = pic_seconds * _LAYOUT_.VIDEO_FPS                          #一个画面总帧数
                
                fileName = getFileNameOfDay(self.prodType, day)
                if not(os.path.exists(fileName)):
                    err_count += 1
                    continue
                if b_flow == True and len(self.features)<1:                         #要素名称在过程中补清除，恢复
                    self.features = ['flow']
                    
                obj = NcData(fileName, self.features, datasetType, self.areaScope, self.depth, self.grid_size)
                datasets = obj.getDataset()
                
                obj = RenderArea(self.features, self.areaScope, datasets, self.grid_size)
                obj.setTitleOfMap(self.features[0],day)
                data, _ = obj.RenderAreaDataset(user)
            
                buffer = io.BytesIO()
                buffer.write(data)  
                img = Image.open(buffer)
                img = np.asarray(img) 
                im = img[:,:,0:3]
                
                while(frames > 0):
                    videoWriter.write(im)
                    frames -= 1            
                
            videoWriter.release()        
            return video_name
         
        except Exception as e:
            print("执行 %s 函数发生错误：%s"  % (sys._getframe().f_code.co_name, e))
            return False     
 
    #生成断面动态文件       
    def create_video_scalar_section(self,user):
        try:
            depthList = []
            pic_seconds = pic_seconds =  _DAT_.PIC_SECONDS  #get_user_config('MISC', 'PIC_SECONDS', user)            #每个画面停留时长
            
            err_count = 0
            datasetType = 'DatasetofSection'
            section_line = self.areaScope
            video_name, videoWriter = self.create_video_writer(user)
            
            all_days = self.get_day_list()
            for day in all_days:
                frames = pic_seconds * _LAYOUT_.VIDEO_FPS                          #一个画面总帧数
                fileName = getFileNameOfDay(self.prodType, day)
                if not(os.path.exists(fileName)):
                    err_count += 1
                    continue
                
                obj = NcData(fileName, self.features, datasetType, section_line, self.depth, self.grid_size)
                datasets = obj.getDataset()
                dataset = datasets[self.features[0]]        #只取一个数据集
                feature = self.features[0]
                
                if depthList == []:
                    if self.prodType == 'CA':
                        depthList, _, dataset = getSectionMaxDepth(dataset, _DAT_.CA_DEPTH_LIST)                          
                    else:
                        depthList, _, dataset = getSectionMaxDepth(dataset, _DAT_.RA_DEPTH_LIST)
                else:
                    dataset = dataset[:len(depthList)]
                    
                obj = RenderSection(feature, section_line, depthList, dataset, self.grid_size)
                obj.setTitleOfMap(feature, day)
                data, _ = obj.RenderSectionDataset(user)
               
                '''图片流变成矩阵'''
                buffer = io.BytesIO()
                buffer.write(data)  
                img = Image.open(buffer)
                img = np.asarray(img) 
                im = img[:,:,0:3]
                
                while(frames > 0):
                    videoWriter.write(im)
                    frames -= 1   
            videoWriter.release()  
                  
            return video_name
         
        except Exception as e:
            print("执行 %s 函数发生错误：%s"  % (sys._getframe().f_code.co_name, e))
            return False
    
#系列垂直剖面线绘制类
class RenderVline():
    def __init__(self, dataset, point, user = 'guest'):
        features = []        
        for key in dataset: 
            features.append(key)
            if isinstance(dataset[key], list):
                dataset[key] = np.array(dataset[key])
        
        depth = dataset['depth']
        dataset.pop('depth')
        features.remove('depth')
                
        if isinstance(depth, list):
            depth = np.array(depth)    
        self.depth = -depth                                     #水深列表，转为负数，用于做数值位置，以便0值最上方
        self.features = features 
        self.dataset = dataset
        self.point = point
        
        read_SYS_USER_option('MAPSIZE', user)
        self.figsize=(MAPSIZE.FIG_WIDTH/2, MAPSIZE.FIG_HEIGHT)
                
    #数据合理性检查，修改依据要素检查数据集，2019-11-9
    def check_data_vaild(self):
        #for key in self.features:
            
        for key in self.dataset:            
            if is_invalid_array(self.dataset[key]):
                return err._NO_VALID_DATA
            
        return True
                  
    #设置图的标题
    def set_map_title(self):
        title_str = ''
        for key in self.features:
            if len(title_str)<1:
                title_str = _DAT_.FEATURE_CHI_NAME[key] + '(' + _DAT_.FEATURE_UNIT[key] +')'
            else:
                title_str = title_str + '、' + _DAT_.FEATURE_CHI_NAME[key] + '(' + _DAT_.FEATURE_UNIT[key] +')'
                 
        #point_str = self.set_point_label()
        #title_str = title_str + '(' + point_str + ')' 
            
        return title_str
   
    #修改此段，使标注数据规范
    def set_ylabel(self):
        mindepth = -self.depth[0]
        maxdepth = -self.depth[-1]
        nums = _LAYOUT_.VLINE_LAB_Y
        values = np.array([2000,1500,1200,1000,800,600,500,400,300,250,200,150,100,80,60,50,40,30,20,15,10,5])
        
        diff = maxdepth - mindepth
        interval = int(diff/nums)   #间隔值
        if interval < 5:          #间隔太小，不进行标注
            return [],[]
        
        iPos = len(values[values > interval])
        interval = values[iPos]
        
#         if mindepth%interval >0:
#             startPos = math.ceil(mindepth/interval) * interval
#             if startPos-mindepth < interval/4:
#                 startPos += interval 
#         else:
#             startPos = mindepth
        startPos = 0
        
        if maxdepth%interval>0:
            endPos = (math.ceil(maxdepth/interval)-1) * interval
            if maxdepth - endPos < interval/4:
                endPos -= interval
        else:
            endPos = maxdepth
            
        steps = int((endPos - startPos)/interval+0.1) + 1
        ylist = np.linspace(startPos, endPos, steps).astype(np.int)
        ylabel_pos = -ylist
        
        ####################################################
        #  minorstep
        x = int(np.round(10 ** (np.log10(interval) % 1)))
        if x in [1, 5, 10]:
            ndivs = 5
        else:
            ndivs = 4
        
        minorstep = interval / ndivs
    
        return ylabel_pos, ylist, minorstep
   
    #点经纬度标注文字
    def set_point_label(self):
        lat = float(self.point['lat'])
        lon = float(self.point['lon'])
        
        if lat>0:
            latunit='°N'
        if lat<0:
            latunit='°S'
        if lat==0:
            latunit='°'                
        
        if lon>0 and lon <180:
            lonunit='°E'
        elif lon<0 or lon >=180:
            lonunit='°W'
        elif lon==0:
            lonunit='°'
        point_str = str(abs(lat))+latunit+','+str(abs(lon))+lonunit
        
        return point_str
       
    #调整图例在图中的位置，防止与图形内容叠加遮挡
    def legend_adjust(self, legend, ax=None, pad=0.05):
        if ax == None: 
            ax = plt.gca()
            
        ax.figure.canvas.draw()
        bbox = legend.get_window_extent().transformed(ax.transAxes.inverted() )
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(-(ymax-ymin)*(1.+pad+bbox.y0), ymax)
        
        return True
    
    def set_map_size(self, width, height):
        self.figsize=(width, height)
        
        return True
        
    #创建叠加模式的垂直剖面线图
    def create_overlay_plotmap(self):
        try:
            res = self.check_data_vaild()
            if res != True:
                return res
            
            fig, ax1 = plt.subplots(figsize = self.figsize) #(MAPSIZE.FIG_WIDTH/2, MAPSIZE.FIG_HEIGHT)
            ax1.set_facecolor(_MAP_.FACECOLOR)
            
            if len(self.features) == 1:
                data = self.dataset[self.features[0]]
                label_str = _DAT_.FEATURE_CHI_NAME[self.features[0]] if self.features[0] in _DAT_.FEATURE_CHI_NAME else 'None'
                ax1.plot(data, self.depth, color = 'blue', linewidth=1, label = label_str)
                
                ax1.tick_params(axis='x', color='blue', which='both')
                ax1.xaxis.set_minor_locator( AutoMinorLocator() )
                ax1.xaxis.set_ticks_position('top')
                    
            else:                                                                           #最多两个要素 
                data1 = self.dataset[self.features[0]]  
                label1 = _DAT_.FEATURE_CHI_NAME[self.features[0]] if self.features[0] in _DAT_.FEATURE_CHI_NAME else 'None'         
                ax1.plot(data1, self.depth, color = 'blue', linewidth=1, label = label1)
                ax1.tick_params(axis='x', labelcolor ='blue', which='both')
                ax1.xaxis.set_minor_locator( AutoMinorLocator() )
                ax1.xaxis.set_ticks_position('top')
                
                data2 = self.dataset[self.features[1]]                                     
                label2 = _DAT_.FEATURE_CHI_NAME[self.features[1]] if self.features[1] in _DAT_.FEATURE_CHI_NAME else 'None'
                ax2 = ax1.twiny()
                ax2.plot(data2, self.depth, color = 'green', linewidth=1, label = label2)
                ax2.tick_params(axis='x', labelcolor ='green', which='both')                                                   
                ax2.xaxis.set_minor_locator( AutoMinorLocator() )
                ax2.xaxis.set_ticks_position('top')
                
            #########################################################                       #XY轴标注    
            ylabel_pos, ylist, minorstep = self.set_ylabel()
            ax1.set_yticklabels(labels=ylist)#, minor = True
            ax1.set_yticks(ylabel_pos)
            ax1.yaxis.set_minor_locator(MultipleLocator(minorstep))
            plt.ylabel("深度 (米)")
            
            title_str = self.set_map_title()
            plt.title(title_str + '\n\n') 
            
            if len(self.features)>1:                                                        #只有一个要素，不绘图例
                leg = fig.legend(loc='upper left',  bbox_to_anchor=(0.65, 0.12), bbox_transform=ax1.transAxes)
                self.legend_adjust(leg)                                                     #将图形下层区域向上提升，留给图例
            
            plt.tight_layout()
            canvas = fig.canvas
            buffer = io.BytesIO()
            canvas.print_png(buffer)
             
            data = buffer.getvalue()
            buffer.close()
            return data
    
        except Exception as e:
            print("执行 %s 函数发生错误：%s"  % (sys._getframe().f_code.co_name, e))
            return False 
    
    #并列模式绘制多幅垂直剖面线图
    def create_para_Vline_map(self):
        try:
            num_feature = len(self.features)
            if num_feature == 1:
                return self.create_overlay_plotmap()
            
            res = self.check_data_vaild()
            if res != True:
                return res

            if num_feature>1:
                fig_width = MAPSIZE.FIG_WIDTH * num_feature/2.
                self.figsize = (fig_width, MAPSIZE.FIG_HEIGHT)   

            fig, axs = plt.subplots(1, num_feature, figsize = self.figsize, sharey=True)
            for i, feature in enumerate(self.features):
                axs[i].plot(self.dataset[feature], self.depth)
                xlabel_str = _DAT_.FEATURE_CHI_NAME[feature] if feature in _DAT_.FEATURE_CHI_NAME else 'None'
                
                axs[i].xaxis.set_ticks_position('top')
                axs[i].xaxis.set_minor_locator(AutoMinorLocator())
                axs[i].set_title(xlabel_str + '\n\n')
                
            ylabel_pos, ylist, minorstep= self.set_ylabel()
            axs[0].set_yticklabels(labels=ylist)#, minor = True
            axs[0].set_yticks(ylabel_pos)
            axs[0].yaxis.set_minor_locator(MultipleLocator(minorstep))
            axs[0].set_ylabel('深度 (米)')
            
            canvas = fig.canvas
            buffer = io.BytesIO()
            canvas.print_png(buffer)
            
            data = buffer.getvalue()
            buffer.close()
            return data        
        
        except Exception as e:
            print("执行 %s 函数发生错误：%s"  % (sys._getframe().f_code.co_name, e))
            return False 

    # 绘制单点垂直剖面箭头流向图
    def create_current_quiver(self):       
        try:
            res = self.check_data_vaild()
            if res != True:
                return res
            
            if 'UVEL' not in self.dataset or 'VVEL' not in self.dataset:
                print('缺少绘制单点流向垂向剖面图所需的要素数据')
                return err._NO_DATA_
            
            uSpeed = np.array(self.dataset['UVEL'])
            vSpeed = np.array(self.dataset['VVEL'])
            uSpeed = np.reshape(uSpeed,(-1,1))
            vSpeed = np.reshape(vSpeed,(-1,1))
            speed = np.sqrt(uSpeed* uSpeed + vSpeed * vSpeed)
            maxSpeed = np.max(speed)
            
            #print(maxSpeed)
            fig, ax = plt.subplots(figsize=self.figsize)            #(MAPSIZE.FIG_WIDTH/2, MAPSIZE.FIG_HEIGHT)
            ax.plot([0,0], [self.depth[0], self.depth[-1]], color = 'black')
            
            q = plt.quiver(0, self.depth, uSpeed*0.3, vSpeed*0.3, scale=0.3, zorder=1000)#np.zeros(len(self.depth))
            
            #绘制单位箭头及说明文字
            str_lab = '最大速度\n' + str(round(maxSpeed,2)) +' 米/秒'
            plt.quiverkey(q, X=0.8, Y=0.05, U = maxSpeed*0.3, label= str_lab, coordinates ='axes', labelpos='N')
            
            point_str = self.set_point_label()
            plt.xticks([0], [point_str])
            
            ylabel_pos, ylist, minorstep= self.set_ylabel()
            ax.set_yticklabels(labels=ylist)#, minor = True
            ax.set_yticks(ylabel_pos)
            ax.yaxis.set_minor_locator(MultipleLocator(minorstep))
            ax.set_ylabel('深度 (米)')
            
            bottom, top = plt.ylim()
            plt.ylim(top = top + 1 * minorstep , bottom = bottom - 1 * minorstep)
            
            #ax.set_ylim(-(ymax-ymin)*(1.+pad+bbox.y0), ymax)    
            
            point_str = self.set_point_label()
            title_str = '垂直剖面流速图\n' 
            plt.title(title_str)
            
            #rect = [0., 0.2, 1., 1.]
            plt.tight_layout()#rect
            canvas = fig.canvas
            buffer = io.BytesIO()
            canvas.print_png(buffer)
            data=buffer.getvalue()
            buffer.close()  
            return data
        
        except Exception as e:
            print("执行 %s 函数发生错误：%s"  % (sys._getframe().f_code.co_name, e))
            return False
        
    #绘制跃层图形
    def create_cline_map(self, cline):         
        try:        
            cline_title = {'THETA':'温跃层', 'SALTanom':'盐跃层', 'rho':'密度跃层','sspd':'声速跃层'}
            
            res = self.check_data_vaild()
            if res != True:
                return res
                
            feature = self.features[0]                                           #只取最后一件
            data = self.dataset[feature]
            min_v = np.min(data)
            max_v = np.max(data)
            
            fig, ax = plt.subplots(figsize=self.figsize)
            plt.plot(data, self.depth, color = 'black')
            
            ############################################################################
            # 上层跃层范围及文字
            mindepth_therm_1 = -cline['mindepth_therm_1']
            maxdepth_therm_1 = -cline['maxdepth_therm_1']
            strength_therm_1 = cline['strength_therm_1']
            
            plt.hlines(maxdepth_therm_1, min_v, max_v, color="lightgray")     #横线
            plt.hlines(mindepth_therm_1, min_v, max_v, color="lightgray")     #横线
            plt.fill_between(np.linspace(min_v, max_v, int((max_v-min_v)/0.5)), maxdepth_therm_1, mindepth_therm_1, facecolor="lightgreen")
            dh = abs(maxdepth_therm_1 - mindepth_therm_1)
            # 跃层的文字和上下箭头
            t=( '上界深度 = '+str(abs(mindepth_therm_1)) + '\n' +
                '厚    度 = ' + str(dh)+  '\n' +
                '强    度 = ' + str(strength_therm_1))
            plt.annotate("", xy=((min_v*3 + max_v)/4, mindepth_therm_1), xytext=((min_v*3 + max_v)/4, maxdepth_therm_1), arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
            plt.text((min_v*3 + max_v)/4, (maxdepth_therm_1 + mindepth_therm_1)/2, t,
                     {'color': 'black', 'fontsize': 8, 'ha': 'left', 'va': 'center',
                       'bbox': dict(boxstyle="round", fc="lemonchiffon", ec="black", pad=0.5)})
     
            ############################################################################
            #存在下层跃层范围及文字
            if 'mindepth_therm_2' in cline:
                mindepth_therm_2 = -cline['mindepth_therm_2']
                maxdepth_therm_2 = -cline['maxdepth_therm_2']
                strength_therm_2 = cline['strength_therm_2']
             
                plt.hlines(maxdepth_therm_2, min_v, max_v, color="lightgray")     #横线
                plt.hlines(mindepth_therm_2, min_v, max_v, color="lightgray")     #横线
                plt.fill_between(np.linspace(min_v, max_v, int((max_v-min_v)/0.5)), maxdepth_therm_2, mindepth_therm_2, facecolor="lightgreen")
                dh = abs(maxdepth_therm_2 - mindepth_therm_2)
                # 跃层的文字和上下箭头
                t=( '上界深度 = '+str(abs(mindepth_therm_2)) + '\n' +
                    '厚    度 = ' + str(dh)+  '\n' +
                    '强    度 = ' + str(strength_therm_2))
                plt.annotate("", xy=((min_v*3 + max_v)/4, mindepth_therm_2), xytext=((min_v*3 + max_v)/4, maxdepth_therm_2), arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
                plt.text((min_v*3 + max_v)/4, (maxdepth_therm_2 + mindepth_therm_2)/2, t,
                         {'color': 'black', 'fontsize': 8, 'ha': 'left', 'va': 'center',
                           'bbox': dict(boxstyle="round", fc="lemonchiffon", ec="black", pad=0.5)})
            
            point_str = self.set_point_label()
            
            plt.title('%s(%s)\n\n' % (cline_title[feature], point_str))
            ax.xaxis.set_ticks_position('top') 
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            
            ylabel_pos, ylist, minorstep = self.set_ylabel()
            ax.set_yticklabels(labels=ylist)#, minor = True
            ax.set_yticks(ylabel_pos)
            ax.yaxis.set_minor_locator(MultipleLocator(minorstep))
            plt.ylabel('深度 (米)')
     
            plt.tight_layout()
       
            canvas = fig.canvas
            buffer = io.BytesIO()
            canvas.print_png(buffer)
            data=buffer.getvalue()
            buffer.close()
    
            return data
        
        except Exception as e:
            print("执行 %s 函数发生错误：%s"  % (sys._getframe().f_code.co_name, e))
            return False 

# 创建定点时序图
def create_timeseries_map(dataset, user = 'guest',correlation_Index = None):
    try: 
        #分别对应：'solid', 'dotted', 'dashed', 'dashdot' ,'densely dashdotdotted'
        #         'dashdotted', 'loosely dashdotted', 'densely dashed', 'loosely dashed', 'loosely dotted' 
        linestyle_str = [(0, ()),    (0, (1, 1)),   (0, (5, 5)),    (0, (10, 2, 1, 2)), (0, (3, 1, 1, 1, 1, 1)),
                         (0, (3,5,1,5)),    (0, (3,10,1,10)),   (0, (5, 1)),    (0, (5, 10)),   (0, (1, 10))]       #9种线型，对应最多9种数值（如深度值）
        line_color = ['green', 'blue']                                                                              #2种颜色，对应最多2类要素
        
        features= []
        max_list = -999
        
        if 'timelist' not in dataset:
            print('传入参数有误，缺少指定键名：timelist的时间列表')
            return False
        
        timelist = dataset['timelist']
        dataset.pop('timelist')
        
        for key in dataset:
            features.append(key)
            
        if isinstance(timelist, list):
            timelist = np.array(timelist)        
        
        read_SYS_USER_option('MAPSIZE', user)   
        fig1, ax1 = plt.subplots(figsize=(MAPSIZE.FIG_WIDTH, MAPSIZE.FIG_HEIGHT))                                      
        
        i, j = 0, 0
        new_features = split_features(features)         #,main_names
                    
        if len(new_features) > 1:
            ax2 = ax1.twinx()
        
        feature_index = 0   
        for feature in new_features:
            if feature_index == 0:
                ax_temp = ax1
            else:
                ax_temp = ax2
            
            label = new_features[feature]['label']
            ax_temp.set_ylabel(label, color = line_color[feature_index])            
            
            if 'feature_values' in new_features[feature]:                   #一个要素有多条时序线的情况，如盐度不同水深时序值
                feature_value_index = 0

                for feature_value in new_features[feature]['feature_values']:
                    #feature_name = new_features[feature]['feature_values'][feature_value]
                    datelist = dataset[feature_value]
                    if isinstance(datelist, list):
                        datelist = np.array(datelist)
                    
                    legend = new_features[feature]['legends'][feature_value_index]
                     
                    ax_temp.plot(timelist, datelist, color=line_color[feature_index], linestyle = linestyle_str[feature_value_index], label = legend)
                    feature_value_index += 1
                    if feature_value_index > 8:
                        break                           #同一要素最多画10条线
            else:                                       #普通类型时序图
                datelist = dataset[feature]
                legend = new_features[feature]['legend']
                ax_temp.plot(timelist, datelist, color=line_color[feature_index], linestyle = linestyle_str[feature_index], label = legend)
                
            feature_index += 1      
            
            if feature_index >1:                        #最多处理两种类型要素
                break  
        
        ax1.set_xlabel('时间')                
             
        if len(features)>1:
            fig1.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
        
        if correlation_Index is not None:       #绘制相关系数
            plt.text(0, max_list*1, 'R=%3f' % correlation_Index, fontsize=14)
            
        if len(timelist)>=8: #如果时间列表大于8个单位，
            interval = int(len(timelist)/8)
            x_major_locator=MultipleLocator(interval)
            ax1.xaxis.set_major_locator(x_major_locator)
            
        for label in ax1.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment('right')
                    
        plt.tight_layout()
        canvas = fig1.canvas #. plt.get_current_fig_manager().canvas    
        
        buffer = io.BytesIO()
        canvas.print_png(buffer)
        data = buffer.getvalue()
        buffer.close()

        return data
    
    except Exception as e:
            print("执行 %s 函数发生错误：%s"  % (sys._getframe().f_code.co_name, e))
            return False 

#气候指数时间序列图
def create_climateindex_map(dataset, index_name, user = 'guest'):
    #read_SYS_USER_option(['_LAYOUT_'])
    
    time_set = dataset['timelist']
    len_time = len(time_set)
    
    read_SYS_USER_option('MAPSIZE', user)
    
    fig = plt.figure(figsize = (MAPSIZE.FIG_WIDTH, MAPSIZE.FIG_HEIGHT), dpi=MAPSIZE.DPI)
    plt.plot(time_set, dataset['climate_index'], color='b', linewidth=2)
    
    plt.axhline(0, color='k')
    plt.title('气候指数')
    plt.xlabel('时间')
    plt.ylabel(index_name)
    plt.xlim(time_set[0], time_set[len_time-1])
    
    canvas = fig.canvas #. plt.get_current_fig_manager().canvas    
    
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    data = buffer.getvalue()
    buffer.close()
    return data
   
#Matplotlib输出图边框范围探测
#输入参数：original_img，内存图像、label图类型。
#    original_img=np.load("E:\\new_project\成果20190812\code\\app\data\\section_img0919.npy")
#    original_img=original_img[:,:,0:3]
def Frame_detecting(original_img,label):
    #RGB转灰度图
    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)      #三波段转为一个波段
    
    #二值化图像，255为白色，0为黑色
    if label=="Area":
        _,img=cv2.threshold(original_img,0,255,cv2.THRESH_BINARY)  
    if label=="Section":
        _,img=cv2.threshold(original_img,27,255,cv2.THRESH_BINARY)
         
    u,v=img.shape
    horizontal=[]
    vertical=[]

    '''判断上下边距'''
    for i in range(0,u,1):
        num = str(img[i:i+1,:].tolist()).count("0")
        horizontal.append(num)
    
    sort1=sorted(horizontal,reverse = True)
    
    if sort1[0]==sort1[1]:
        idx=[i for i,x in enumerate(horizontal) if x==sort1[0]]
        top_padding=idx[0] 
        bottom_padding=idx[1]      
    
    else:  
        idx1=horizontal.index(sort1[0])
        idx2=horizontal.index(sort1[1])
        if idx1>idx2:
            top_padding=idx2 
            bottom_padding=idx1
        
        else:
            top_padding=idx1
            bottom_padding=idx2
            
    '''判断左右边距'''#因为存在色标，有四个差不多的最大值，按顺序排列后，前2个分为边距
    for j in range(0,v,1):
        num1 = str(img[:,j:j+1].tolist()).count("0")
        vertical.append(num1)

    sort2=sorted(vertical,reverse = True)
    
    idx1=[i for i,x in enumerate(vertical) if x==sort2[0]] 
    idx2=[i for i,x in enumerate(vertical) if x==sort2[1]] 
    idx3=[i for i,x in enumerate(vertical) if x==sort2[2]] 
    idx4=[i for i,x in enumerate(vertical) if x==sort2[3]] 
    
        
    idx=idx1+idx2+idx3+idx4  
    idx=sorted(idx,reverse = False)   
        
    left_padding=idx[0]
    index = 1
    while(idx[index] == left_padding):
        index += 1
    right_padding = idx[index]
    
    return top_padding,bottom_padding,left_padding,right_padding