# -*- coding: utf-8 -*-
import psycopg2

#从数据库创建一个表
def connectPostgreSQL():
    conn = psycopg2.connect(database = 'postgres', user='postgres',password='123liuchao123',host='localhost',port='5432')
    cursor=conn.cursor()
    cursor.execute('''create table public.Argo_sql2017(
    ID INT PRIMARY KEY NOT NULL,
    LAT DECIMAL NOT NULL,
    LON DECIMAL NOT NULL,
    PRES DECIMAL NOT NULL,
    TEMP DECIMAL NOT NULL,
    SALT DECIMAL NOT NULL,
    MLD DECIMAL NOT NULL,
    ILD DECIMAL NOT NULL,
    SSH DECIMAL NOT NULL,
    SST DECIMAL NOT NULL,
    MONTH DECIMAL NOT NULL,
    YEAR DECIMAL NOT NULL)''')
    conn.commit()
    conn.close()
    print('table public.Argo_sql is created!')

#在创建的表里插入数据
def insertOperate(dataset):
    num=dataset['id']
    argo_lat=float(dataset['lat'])
    argo_lon=float(dataset['lon'])
    argo_pres=float(dataset['pres'])
    argo_temp=float(dataset['temp'])
    if argo_temp != argo_temp:
        argo_temp = 9999
    argo_salt=float(dataset['salt'])
    if argo_salt != argo_salt:
        argo_salt = 9999
    argo_mld=float(dataset['mld'])
    if argo_mld != argo_mld:
        argo_mld = 9999
    argo_ild=float(dataset['ild'])
    if argo_ild != argo_ild:
        argo_ild = 9999
    ssh = float(dataset['SSH'])
    if ssh != ssh:
        ssh = 9999
    sst = float(dataset['SST'])
    if sst != sst:
        sst = 9999
    month = float(dataset['month'])
    year = float(dataset['year'])
    
    conn = psycopg2.connect(database='postgres',user='postgres',password='123liuchao123',host='localhost',port='5432')
    cursor = conn.cursor()
    
    cursor.execute('''insert into public.Argo_sql2017(ID,LAT,LON,PRES,TEMP,SALT,MLD,ILD,SSH,SST,MONTH,YEAR)
        values(%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f)'''%(num,argo_lat,argo_lon,argo_pres,argo_temp,argo_salt,argo_mld,argo_ild,ssh,sst,month,year))
    
    conn.commit()

    conn.close()
    print('insert records into public.wod_sqlA successfully')

#从数据库里选取某组数据，判断这组数据是否在数据库中
def selectOperate(id_count=0,type='equal',areaScope=None,month=None,year=None):
    conn = psycopg2.connect(database = 'postgres', user='postgres',password='123liuchao123',host='localhost',port='5432')
    cursor=conn.cursor()
    #取该ID的数据
    if type == 'equal':
        cursor.execute("select ID,LAT,LON,PRES,TEMP,SALT,MLD,ILD,SSH,SST,MONTH,YEAR from public.Argo_sql3 where MONTH=%d and YEAR=%d and lat<=%d and lon<=%d and temp != 9999 and salt!=9999 and SST!=9999 and SSH!=9999 limit %d"%(month,year,30,130,id_count))
    #取小于该ID的所有值
    elif type == 'max':
        cursor.execute("select ID,LAT,LON,PRES,TEMP,SALT,MLD,ILD,SSH,SST,MONTH,YEAR from public.Argo_sql where id<%d" % (id_count))
    # 取大于该ID的所有值
    elif type == 'min':
        cursor.execute("select ID,LAT,LON,PRES,TEMP,SALT,MLD,ILD,SSH,SST,MONTH,YEAR from public.Argo_sql where id>=%d" % (id_count))
    #获取WOD数据
    elif type == 'wod':
        minLat = areaScope['minlat']
        maxLat = areaScope['maxlat']
        minLon = areaScope['minlon']
        maxLon = areaScope['maxlon']
        cursor.execute("select LAT,LON,DEPTH,TEMP,SALT,SSH,SST,MONTH from public.wod_sql where lat>=%d and lat<=%d and lon>=%d and lon<=%d and month=%d limit %d" % (minLat,maxLat,minLon,maxLon,month,id_count))
    #获取所有数据
    elif type == 'interval':
        id_count_min = id_count[0]
        id_count_max = id_count[1]
        cursor.execute("select ID,LAT,LON,PRES,TEMP,SALT,MLD,ILD,SSH,SST,MONTH,YEAR from public.Argo_sql where id in (%d,%d)" % (id_count_min,id_count_max))
    elif type == 'scope':
        minLat=areaScope['minlat']
        maxLat=areaScope['maxlat']
        minLon=areaScope['minlon']
        maxLon=areaScope['maxlon']
        cursor.execute(
            "select ID,LAT,LON,PRES,TEMP,SALT,MLD,ILD,SSH,SST,MONTH,YEAR from public.Argo_sql where lat>=%d and lat<=%d and lon>=%d and lon<=%d and month=%d and depth <= 500 and salt >= 30 and salt <= 37 limit %d" % (minLat,maxLat,minLon,maxLon,month,id_count))
    rows=cursor.fetchall()
    if len(rows) != 0:
        return rows
    else:
        return None


def selectOperate2017(id_count=0, type='equal', areaScope=None, month=None, year=None):
    conn = psycopg2.connect(database='postgres', user='postgres', password='123liuchao123', host='localhost',
                            port='5432')
    cursor = conn.cursor()
    # 取该ID的数据
    if type == 'equal':
        cursor.execute(
            "select ID,LAT,LON,PRES,TEMP,SALT,MLD,ILD,SSH,SST,MONTH,YEAR from public.Argo_sql2017 where MONTH=%d and YEAR=%d and lat<=%d and lon<=%d and temp != 9999 and salt!=9999 and SST!=9999 and SSH!=9999 limit %d" % (
            month, year, 30, 130,id_count))
    # 取小于该ID的所有值
    rows = cursor.fetchall()
    if len(rows) != 0:
        return rows
    else:
        return None

