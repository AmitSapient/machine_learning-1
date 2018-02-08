

import csv
import DBManager
import datetime
import requests
import EmailUtil
import urllib.request
from io import TextIOWrapper
from zipfile import ZipFile
#from StringIO import StringIO
from io import StringIO
from urllib.request import urlopen
import shutil
import pandas as pd

class NSE_Result_Calendar_Update_Process:

    def __init__(self):
        self.con = DBManager.connectDB()
        self.cur = self.con.cursor()

    def csv_reader(self):
        
        url_prefix = 'https://www.nseindia.com/archives/nsccl/mwpl/nseoi_'
        url_suffix = '.zip'
        start = '2017-02-01'
        end = '2018-02-08'
        daterange = pd.date_range(start, end, freq='B')
        for single_date in daterange:
            date_str = single_date.strftime("%d%m%Y")
            print (date_str)
            url =  url_prefix+date_str+url_suffix
            print ('url - ', url)            
            try:

    #        url = urlopen('https://www.nseindia.com/archives/nsccl/mwpl/nseoi_07022018.zip')
                req = urllib.request.Request(url, headers={'User-Agent' : "Magic Browser"}) 
                con = urllib.request.urlopen( req )
                filename = 'nseoi_'+date_str+url_suffix
                file = open(filename, 'wb')
    #        shutil.copyfileobj(con, filename)
                file.write(con.read())
                
                zf = ZipFile(filename)
                csvfilename = 'nseoi_'+date_str+'.csv'
                df = pd.read_csv(zf.open(csvfilename))            
#                print(df)
                df.to_csv(csvfilename)
            except Exception as e:
                #come here for nationalholiday days
                print('exception for url - ', url)
                continue


        

    def extract_zip(self,input_zip):
        input_zip=ZipFile(input_zip)
        return {name: input_zip.read(name) for name in input_zip.namelist()}


# ----------------------------------------------------------------------
thisObj = NSE_Result_Calendar_Update_Process()
thisObj.csv_reader()



