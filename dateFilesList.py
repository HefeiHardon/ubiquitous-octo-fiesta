# -*- coding: utf-8 -*-

import arrow
import datetime
import time
import os
import numpy as np

_YEAR_      = 0
_MONTH_     = 1
_DAY_       = 2

# 依据统计方法与时间范围，获取相应的日期列表
class DaysList():
    def __init__(self, statTimeInfo, statUnit = 2):
        # statTimeType：时间统计类型，0为累年法统计，1为日期区间法统计
        # minYear:起始年
        # maxYear:终止年
        # minMonth:起始月
        # maxMonth:终止月
        # minDay:起始日
        # maxDay:终止日

        self.timeStatType = int(statTimeInfo['timeStatType'])
        startDay = statTimeInfo['startDay']
        endDay = statTimeInfo['endDay']
        minYear, maxYear, minMonth, maxMonth, minDay, maxDay = getTimeScope(startDay,endDay)
        self.minYear = int(minYear)
        self.maxYear = int(maxYear)
        self.minMonth = int(minMonth)
        self.maxMonth = int(maxMonth)
        self.minDay = int(minDay)
        self.maxDay = int(maxDay)
        self.statUnit = statUnit

    def DaysofInterval(self, year1, year2, month1, month2, day1, day2):
        # 计算两个时间点（日期）之间相邻的天数
        date1 = datetime.date(year=year1, month=month1, day=day1)
        date2 = datetime.date(year=year2, month=month2, day=day2)
        return (date2 - date1).days + 1

    def AllDaysList(self):
        # 按照不同统计类型（0：累年法，1：日期区间法），获取全部日期列表
        if self.timeStatType == 0:
            all_day_list = self.DaysListofAnnualyear()
        else:
            all_day_list = self.DaysListofDayScope()
        return all_day_list

    def DaysListofDayScope(self):
        # 日期区间统计法，获取指定日期区间之内的全部日期列表
        start_date = datetime.date(year=self.minYear, month=self.minMonth, day=self.minDay)
        a = 0
        all_day_list = []
        days_sum = self.DaysofInterval(self.minYear, self.maxYear, self.minMonth, self.maxMonth, self.minDay,
                                       self.maxDay)
        while a < days_sum:
            b = arrow.get(start_date).shift(days=a).format(("YYYYMMDD"))
            a += 1
            all_day_list.append(b)
        return all_day_list

    def DaysListofAnnualyear(self):
        # 累年统计法，获取累年的时间范围内全部日期列表
        all_day_list = []
        for eachYear in range(self.minYear, self.maxYear + 1, 1):
            a = 0
            start_date = datetime.date(year=int(eachYear), month=int(self.minMonth), day=int(self.minDay))
            days_sum = self.DaysofInterval(eachYear, eachYear, self.minMonth, self.maxMonth, self.minDay, self.maxDay)
            while a < days_sum:
                b = arrow.get(start_date).shift(days=a).format(("YYYYMMDD"))
                a += 1
                all_day_list.append(b)
        return all_day_list
    
    #获取全部日期矩阵，主要考虑统计单元（statUnit）因素
    def AllDaysArray(self):
        all_days_array = list()
        
        all_days_list = self.AllDaysList()
        if self.statUnit == _YEAR_:
            for year in range(self.minYear, self.maxYear+1, 1):
                daysList = [day for day in all_days_list if day[:4] == str(year)]
                all_days_array.append(daysList)
                
        elif self.statUnit == _MONTH_:
            yearMonthList = [day[:6] for day in all_days_list]
            yearMonthList = list(set(yearMonthList))
            yearMonthList.sort()
            for month in yearMonthList:
                daysList = [day for day in all_days_list if day[:6] == month]
                all_days_array.append(daysList)
        else:
            all_days_array = all_days_list
        res=np.array(all_days_array)                    
        return res
            
#获取最小、最大年月日
def getTimeScope(startDay,endDay):
    minYear = startDay[:4] 
    minMonth = startDay[4:6]
    minDay = startDay[6:8]
    
    maxYear = endDay[:4] 
    maxMonth = endDay[4:6]
    maxDay = endDay[6:8]
    return minYear, maxYear, minMonth, maxMonth, minDay, maxDay