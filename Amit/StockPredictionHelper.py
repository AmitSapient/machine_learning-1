
import quandl
import pandas as pd
import nsepy as nse
from datetime import date, timedelta
import numpy as np

class StockPredictionHelper:
    
#    def __init__(self):
#        self.result_df = [] 
        
        
    
    def getData(self, ticker, flag):
        
        if flag == 'real_time':
            
            Authkey = '5_pRK9pKefuvZzHe-MkSy'
            
            nse_dataset = "NSE" + "/" + ticker
    #        mydata = quandl.get(nse_dataset, authtoken=Authkey, rows=1000, sort_order="asc")
            mydata = quandl.get(nse_dataset, authtoken=Authkey, start_date='2014-01-01', sort_order="asc")
            
            
            # print mydata
            df = pd.DataFrame(mydata)
#            print (ticker, " first record - \n", df[:1])
            filename = 'datasets\\'+ticker+'.csv'
            
            df.to_csv(filename)
    #        df.columns = ['open', 'high','low','close','volume','turnover']
            df.columns = ['date','open', 'high','low','close','volume','turnover']
            print ('Got data from Quandl - ', ticker)
            
        else:
            df = pd.read_csv("D:\\workspace_pyCharm\\Machine Learning\\datasets\\NIFTY_50.csv")
            df.columns = ['date','open', 'high','low','close','volume','turnover']
            df = df.drop(['date'],1)
            print ('Got data from CSV - ', ticker)
        
    #    print(df.columns)
        #'Open', 'High', 'Low', 'Close', 'Shares Traded', 'Turnover (Rs. Cr)'
        
        #dd = df.to_dict(orient='dict')
        #ddd = dd['Close'] 
        
    #    print (df)
        return df
    
    
    def getNiftyData(self,ticker, flag):
        
        if flag == 'real_time':
            
            Authkey = '5_pRK9pKefuvZzHe-MkSy'
            
            nse_dataset = "NSE" + "/" + ticker
            mydata = quandl.get(nse_dataset, authtoken=Authkey, start_date='2014-01-01', sort_order="asc")
            
            # print mydata
            df = pd.DataFrame(mydata)
            print ("Nifty first record - \n", df[:1])
            filename = 'datasets\\'+ticker+'.csv'
            
            df.to_csv(filename)
            df.columns = ['open', 'high','low','close','volume','turnover']
    #        df.columns = ['date','open', 'high','low','close','volume','turnover']
            print ('Got data from Quandl')
            
        else:
            df = pd.read_csv("D:\\workspace_pyCharm\\Machine Learning\\datasets\\NIFTY_50.csv")
            df.columns = ['date','open', 'high','low','close','volume','turnover']
            df = df.drop(['date'],1)
            print ('Got data from CSV')
         
        return df   
        
    
    def getDataFromNSE(self,ticker, flag):
        
            mydata = nse.get_history(symbol=ticker, 
                        start=date(2017,1,1), 
                        end=date(2017,12,31),
    					index=True)
            df = pd.DataFrame(mydata)
            filename = 'datasets\\'+ticker+'.csv'
            
            df.to_csv(filename)
    #        df.columns = ['open', 'high','low','close','volume','turnover']
            df.columns = ['open', 'high','low','close','volume','turnover']
            print ('Got data from Quandl')
    

    def moving_average(self, df):
        #MSE does not change using 4 below
#        df['1dma'] = df['close'].rolling(window=1).mean()
#        df['2dma'] = df['close'].rolling(window=2).mean()
#        df['3dma'] = df['close'].rolling(window=3).mean()
#        df['4dma'] = df['close'].rolling(window=4).mean()
        df['5dma'] = df['close'].rolling(window=5).mean()
        df['10dma'] = df['close'].rolling(window=10).mean()
        df['15dma'] = df['close'].rolling(window=15).mean()
        df['20dma'] = df['close'].rolling(window=20).mean()
        df['50dma'] = df['close'].rolling(window=50).mean()
        #following two seems to be important as MSE drop using them
        df['100dma'] = df['close'].rolling(window=100).mean()
        df['200dma'] = df['close'].rolling(window=200).mean()

        return df    

    def MACD(self, df):
        
#        df['26ema'] = pd.ewma(df["close"], span=26)
        df['26ema'] = df['close'].ewm(span=26).mean()
#        df['12ema'] = pd.ewma(df["close"], span=12)
        df['12ema'] = df['close'].ewm(span=12).mean()
        
        df['MACD'] = df['12ema'] - df['26ema']
        
        #Now calculate 9 ema signal line
        df['macd_9ema_signal'] = df['MACD'].ewm(span=9).mean()
#        df['macd_signal_diff'] = df['macd_9ema_signal']  - df['MACD']
        df = df.drop(['26ema', '12ema'],1 )
#        df = df.drop(['26ema', '12ema', 'MACD','macd_9ema_signal'],1 )
        
#        df.plot(y = ["MACD"], title = "MACD")
        return df  

    def addMACDAsFeature(self, df):
        
        mydf = self.MACD(df)
        mydf = self.calculateRSI(mydf)
        mydf['macd_0'] = mydf['MACD']
        mydf['macd_1'] = mydf['MACD'].shift(1)
        mydf['macd_2'] = mydf['MACD'].shift(2)
        mydf['macd_sig_0'] = mydf['macd_9ema_signal']       
        mydf['macd_sig_1'] = mydf['macd_9ema_signal'].shift(1)
 
        mydf['rsi0'] = mydf['rsi']
        mydf['rsi1'] = mydf['rsi'].shift(1)
        mydf['rsi2'] = mydf['rsi'].shift(2)
        
#        print ('mydf - ', mydf)
        mydf['diff_0'] = mydf['macd_0']  - mydf['macd_sig_0']
        mydf['diff_1'] = mydf['macd_1']  - mydf['macd_sig_1']
        
        pos_threshold = 0.3
        neg_threshold = -0.3
        '''
        mydf['macd_result'] = np.where((mydf['macd_0'] < 0) & (mydf['diff_0'] < 0 ) & ( mydf['diff_1'] < mydf['diff_0']) & (mydf['diff_0'] > neg_threshold)  , 'MACD-PBuy', 'MACD-PHold')
        mydf['macd_result'] = np.where((mydf['macd_result'] == "MACD-PHold") & (mydf['macd_0'] > 0) & (mydf['diff_0'] > 0 ) & ( mydf['diff_1'] > mydf['diff_0']) & (mydf['diff_0'] < pos_threshold)  , 'MACD-PSell', mydf['macd_result'])    
        mydf['macd_result'] = np.where((mydf['macd_0'] > mydf['macd_sig_0'] ) & (mydf['macd_1'] <= mydf['macd_sig_1'] ), 'MACD-Buy', mydf['macd_result'])    
        mydf['macd_result'] = np.where((mydf['macd_0'] < mydf['macd_sig_0'] ) & (mydf['macd_1'] >= mydf['macd_sig_1'] ), 'MACD-Sell', mydf['macd_result'])    
        
        mydf['rsi_result'] = np.where((mydf['rsi0'] < 70) & (mydf['rsi0'] > 50 ) & (mydf['rsi0'] < mydf['rsi1']) & (mydf['rsi1'] < mydf['rsi2']) & (mydf['rsi2'] > 70 )  , 'RSI-Sell', 'RSI-Hold')
        mydf['rsi_result'] = np.where((mydf['rsi0'] > 70) & (mydf['rsi0'] < mydf['rsi1']) & (mydf['rsi1'] < mydf['rsi2'])  , 'RSI-PSell', mydf['rsi_result'])
        mydf['rsi_result'] = np.where((mydf['rsi0'] > 30) & (mydf['rsi0'] < 50 ) & (mydf['rsi1'] < mydf['rsi0']) & (mydf['rsi2'] < mydf['rsi1']) & (mydf['rsi2'] < 30 )  , 'RSI-Buy', mydf['rsi_result'])
        mydf['rsi_result'] = np.where((mydf['rsi0'] < 30) & (mydf['rsi0'] > 20 ) & (mydf['rsi1'] < mydf['rsi0'])  , 'RSI-PBuy', mydf['rsi_result'])
        '''
        '''
        #sell=-2, Psell=-1, hold=0, pbuy=1, buy=2
        mydf['macd_result'] = np.where((mydf['macd_0'] < 0) & (mydf['diff_0'] < 0 ) & ( mydf['diff_1'] < mydf['diff_0']) & (mydf['diff_0'] > neg_threshold)  ,1, 0)
        mydf['macd_result'] = np.where((mydf['macd_result'] == 0) & (mydf['macd_0'] > 0) & (mydf['diff_0'] > 0 ) & ( mydf['diff_1'] > mydf['diff_0']) & (mydf['diff_0'] < pos_threshold)  , -1, mydf['macd_result'])    
        mydf['macd_result'] = np.where((mydf['macd_0'] > mydf['macd_sig_0'] ) & (mydf['macd_1'] <= mydf['macd_sig_1'] ), 2, mydf['macd_result'])    
        mydf['macd_result'] = np.where((mydf['macd_0'] < mydf['macd_sig_0'] ) & (mydf['macd_1'] >= mydf['macd_sig_1'] ), -2, mydf['macd_result'])    
        
        mydf['rsi_result'] = np.where((mydf['rsi0'] < 70) & (mydf['rsi0'] > 50 ) & (mydf['rsi0'] < mydf['rsi1']) & (mydf['rsi1'] < mydf['rsi2']) & (mydf['rsi2'] > 70 )  , -2, 0)
        mydf['rsi_result'] = np.where((mydf['rsi0'] > 70) & (mydf['rsi0'] < mydf['rsi1']) & (mydf['rsi1'] < mydf['rsi2'])  , -1, mydf['rsi_result'])
        mydf['rsi_result'] = np.where((mydf['rsi0'] > 30) & (mydf['rsi0'] < 50 ) & (mydf['rsi1'] < mydf['rsi0']) & (mydf['rsi2'] < mydf['rsi1']) & (mydf['rsi2'] < 30 )  , 2, mydf['rsi_result'])
        mydf['rsi_result'] = np.where((mydf['rsi0'] < 30) & (mydf['rsi0'] > 20 ) & (mydf['rsi1'] < mydf['rsi0'])  , 1, mydf['rsi_result'])
        '''

        #sell=-2, Psell=-1, hold=0, pbuy=1, buy=2
        mydf['macd_result'] = np.where((mydf['macd_0'] < mydf['macd_1'])  ,-1, 0) #sell
        mydf['macd_result'] = np.where((mydf['macd_0'] > mydf['macd_1']) ,1, mydf['macd_result']) #buy
        
        mydf['rsi_result'] = np.where((mydf['rsi0'] < mydf['rsi1'])   , -1, 0)
        mydf['rsi_result'] = np.where((mydf['rsi0'] > mydf['rsi1'])   , 1, mydf['rsi_result'])


#        print(mydf.columns)
        mydf = mydf.drop(['diff_0', 'diff_1', 'rsi0','rsi1','rsi2','macd_0','macd_1','macd_2','macd_sig_0','macd_sig_1','MACD', 'macd_9ema_signal', 'rsi'],1)
#        print (mydf)
#        print(mydf.columns)
        return mydf
        
        
    def calculateRSI(self, df):
        n=14
        delta = df['close'].diff()
        dUp, dDown = delta.copy(), delta.copy()
        dUp[dUp < 0] = 0
        dDown[dDown > 0] = 0
        
        RolUp = pd.rolling_mean(dUp, n)
        RolDown = pd.rolling_mean(dDown, n).abs()
        
        RS = RolUp / RolDown
        RSI = 100.0 - (100.0 / (1.0 + RS))
#        print("RSI - ", RSI)
        df['rsi'] = RSI
        return df
    
    def predictNextDayPrice(self):
        
        df = self.getData('NIFTY_50')
        
    def addChangeFromPreviousDay(self, df):
        
        df['prev_day_close'] = df['close'].shift(1)
        df['change'] = df['close'] - df['prev_day_close']
        df = df.drop(['prev_day_close'],1 )
        return df
    
    def addNiftyCloseAsFeature(self, df, flag):
        temp = self.getNiftyData("NIFTY_50", flag)
        #Drop all column but 'close'
        temp = temp[['close']]
        temp=temp.rename(columns = {'close':'nifty_close'})
        #concate does an outerjoin.. this is case where some records in Nist proces missing
        # for few dates
        df = pd.concat([df, temp], axis=1)
#        #some nifty values are coming as NaN
#        df = df.fillna( df['nifty_close'].rolling(window=1).mean()) 
        return df
    
    def addLast60DaysClose(self, df, days):
        for i in range(1, days):
            close_diff = 'close_diff_{}'.format(i)
            df[close_diff] = df['close'].shift(i)
            
        return df    