import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math, datetime, time

from sklearn import preprocessing, cross_validation, svm, model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from xgboost import XGBClassifier,XGBRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import GetStockMarketData as stockdata

class Nifty_Regression_Ex:
    def __init__(self):
#        self.result_df = [] 
        self.result_df = pd.DataFrame()
        
        
    def moving_average(self, df):
        df['5dma'] = df['close'].rolling(window=5).mean()
        df['20dma'] = df['close'].rolling(window=20).mean()
        df['50dma'] = df['close'].rolling(window=50).mean()
        df['100dma'] = df['close'].rolling(window=100).mean()
        #df['200dma'] = df['close'].rolling(window=200).mean()

        return df    

    def MACD(self, df):
        
#        df['26ema'] = pd.ewma(df["close"], span=26)
        df['26ema'] = df['close'].ewm(span=26).mean()
#        df['12ema'] = pd.ewma(df["close"], span=12)
        df['12ema'] = df['close'].ewm(span=12).mean()
        
        df['MACD'] = df['12ema'] - df['26ema']
        
        df = df.drop(["26ema", "12ema"],1 )
#        df.plot(y = ["MACD"], title = "MACD")
        return df  
    
    def predictNextDayPrice(self):
        
        df = stockdata.getData('NIFTY_50')
        
        
    def backTesting(self):
        print("\n\n\n\n\n***********  New Run of Nifty_Regression_Ex **********************\n\n")
        pd.set_option('display.expand_frame_repr', False)
        
#        df = pd.read_csv("D:\\workspace_pyCharm\\Machine Learning\\datasets\\NYSE\\GOOG.csv")
        df = stockdata.getData('NIFTY_50')
        
        #fil NaN data with average
        df = df.fillna(df.mean())
#        print ("Top 5 - ", df.head(5))
#        print ("Last 5 - ", df.tail(5))
        
        #Just take some X years of data if not all years
        #df = df[1259:]
        

        
        # Add another column next_day_close exct duplicate of close. Later we shift to make prev close.
        # First shift 'close' by -1 because today's close depends on all the feature values of yesterday. To predict today's close
        # we need the open, high, lo, volumne of yesterday because we dont have these values yet for today ! 
        df['next_day_close'] = df['close'].shift(-1)
        # Add few Moving avgs to dataframe
        df = self.moving_average(df)
        df = self.MACD(df)
        df = df.dropna(axis=0, how='any')          
        df = df[:-1] # remove last record
#        print ("Last 5 records - ", df.tail(5))
        

        # forecast_out = int(math.ceil(0.01*len(df)))
#        forecast_out = 1  # Predict only Next One day
        #Eprint(len(df))
        #print(forecast_out)
 
        
#        X_Validation_Before_PreProcessing = np.array(df['next_day_close'])
#        X_Validation_Before_PreProcessing = X_Validation_Before_PreProcessing[-forecast_out:]
        
#        X = np.array(df.drop(['next_day_close','open','low','high', 'turnover'],1))
        X_df = df.drop(['next_day_close','open','low','high', 'turnover'],1)
        
        #X = np.array(df.drop(['close', 'symbol', 'date'],1))
        print ("DF Columns - ",X_df.columns)
       
        X = np.array(X_df)
        X = preprocessing.scale(X)
#        X = preprocessing.normalize(X)
        
        print ("After preprocessing- ", X[-5:])

#        X_Validation = X[-forecast_out:]

#        X = X[:-forecast_out]

        y = np.array(df['next_day_close'])
#        y = y[:-forecast_out]
        print ("Y values - ", y[-5:])

        print("Dataset lenghts - ",(len(X), len(y)))


        #X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)
        X_train, X_rest, y_train, y_rest = model_selection.train_test_split(X, y, test_size=0.4, random_state=42)
        X_test, X_Validation, y_test, y_Validation = model_selection.train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)
        
        print("Dataset lenghts Train set- ",(len(X_train), len(y_train)))
        print("Dataset lenghts Test set- ",(len(X_test), len(y_test)))
        print("Dataset lenghts validation set- ",(len(X_Validation), len(y_Validation)))
        
        #result_df = pd.DataFrame()
#        result_df.columns = [['Actual Close', 'Linear Reg', 'SVM', 'Decision Tree']]
#        self.result_df.loc[0, 'Actual Close'] = X_Validation[0]

#        df.loc[df.shape[0]] = ['d', 3] 
        
        
        self.callLinearRegression(X_train, X_test, y_train, y_test,X_Validation, y_Validation )
#        self.callLogisticRegression(X_train, X_test, y_train, y_test,X_Validation )
#        self.callSVM(X_train, X_test, y_train, y_test,X_Validation , y_Validation)
        self.callDecisionTreeRegressor(X_train, X_test, y_train, y_test,X_Validation , y_Validation)
        self.callRandomForest(X_train, X_test, y_train, y_test,X_Validation , y_Validation)
#        self.callGradientBoosting(X_train, X_test, y_train, y_test,X_Validation , y_Validation)
#        self.callXGBClassifier(X_train, X_test, y_train, y_test,X_Validation , y_Validation)
#        self.callKNN(X_train, X_test, y_train, y_test,X_Validation , y_Validation)
    
        print (self.result_df)
        
    def callLinearRegression(self, X_train, X_test, y_train, y_test,X_Validation , y_Validation):
        # Linear regression
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print ("\n\n****** LinearRegression Accuracy  - ",accuracy)
        y_predicted = clf.predict(X_Validation)

        print ("\ny_Validation - ", y_Validation )
        print ("\ny_predicted - ", y_predicted )
        print("\n Diff  - ", y_Validation - y_predicted )
        
#        plt.scatter(y_Validation, y_predicted)
#        plt.xlabel('True Values')
#        plt.ylabel('Predictions')

        mse = mean_squared_error(y_Validation, y_predicted)
        variance = np.var(y_predicted) 
        
        print ("Varinace - ", variance)
        
        
    def callDecisionTreeRegressor(self, X_train, X_test, y_train, y_test,X_Validation , y_Validation):
        #Decision Tree
        clf = tree.DecisionTreeRegressor()
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print ("\n\n****** Decision Tree Accuracy  - ",accuracy)
        y_predicted = clf.predict(X_Validation)
        print ("\ny_Validation - ", y_Validation )
        print ("\ny_predicted - ", y_predicted )
        print("\n Diff  - ", y_Validation - y_predicted )

#        self.result_df.loc[0, 'DecisionTree'] = forecast_set  

    def callRandomForest(self, X_train, X_test, y_train, y_test,X_Validation , y_Validation ):
        #Random forest
        clf = RandomForestRegressor(n_estimators=20)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print ("\n\n****** RandomForestClassifier Accuracy  - ",accuracy)
        y_predicted = clf.predict(X_Validation)
        print ("\ny_Validation - ", y_Validation )
        print ("\ny_predicted - ", y_predicted )
        print("\n Diff  - ", y_Validation - y_predicted )

#        self.result_df.loc[0, 'RandomForest'] = forecast_set 
        
    def callGradientBoosting(self, X_train, X_test, y_train, y_test,X_Validation , y_Validation):
        #GradientBoostingClassifier
        clf= GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print ("\n\n****** GradientBoostingRegressor Accuracy  - ",accuracy)
        forecast_set = clf.predict(X_Validation)
#        print ("X_Validation_Before_PreProcessing - ")
#        print ("GradientBoostingRegressor forecast Values - ", forecast_set)
        self.result_df.loc[0, 'GradientBoosting'] = forecast_set 
        

    def callKNN(self, X_train, X_test, y_train, y_test,X_Validation , y_Validation):
         #KNN
        clf = KNeighborsRegressor()
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
#        print ("\n\n****** KNeighborsRegressor Tree Accuracy  - ",accuracy)
        forecast_set = clf.predict(X_Validation)
#        print ("X_Validation_Before_PreProcessing - ")
#        print ("KNN forecast Values - ", forecast_set)
        self.result_df.loc[0, 'KNN'] = forecast_set 
        
        # self.plot_graph(forecast_set, df)
        
    def callXGBClassifier(self, X_train, X_test, y_train, y_test,X_Validation , y_Validation):
        #XGBClassifier
        clf = XGBRegressor()
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
#        print ("\n\n****** XGBClassifier Accuracy  - ",accuracy)     
        forecast_set = clf.predict(X_Validation)
#        print ("\nX_Validation_Before_PreProcessing - ")
#        print (" XGBRegressor forecast Values - ", forecast_set)
        self.result_df.loc[0, 'XGBRegressor'] = forecast_set 
        

    def callLogisticRegression(self, X_train, X_test, y_train, y_test,X_Validation , y_Validation ):
        # Linear regression
        clf = LogisticRegression(penalty='l2',C=.01)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
#        print ("\n\n****** LogisticRegression Accuracy  - ",accuracy)
        forecast_set = clf.predict(X_Validation)
#        print ("X_Validation_Before_PreProcessing - ")
#        print ("LogisticRegression forecast Values - ", forecast_set)
        self.result_df.loc[0, 'Logistic Reg'] = forecast_set  
        
    def callSVM(self, X_train, X_test, y_train, y_test,X_Validation , y_Validation ):
        # SVM        
        clf = svm.SVR(kernel='poly')
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
#        print ("\n\n****** SVM Accuracy  - ",accuracy)
        forecast_set = clf.predict(X_Validation)
#        print ("X_Validation_Before_PreProcessing - ")
#        print ("SVM forecast Values - ", forecast_set)
        self.result_df.loc[0, 'SVM'] = forecast_set  

    def plot_graph(self,forecast_set, df):
        df['Forecast'] = np.nan
        last_date = df.iloc[-1].date
        last_date = datetime.datetime.strptime(last_date, "%m/%d/%Y")
        # last_unix = last_date.timestamp()
        last_unix = time.mktime(last_date.timetuple())
        #last_unix = last_date
        one_day = 86400
        next_unix = last_unix + one_day

        for i in forecast_set:
            next_date = datetime.datetime.fromtimestamp(next_unix)
            next_unix += one_day
            df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
        print (df.head(n=10))
        print (df.tail(n=10))
        df['close'].plot()
        df['Forecast'].plot()
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()




if __name__ == "__main__":
    c = Nifty_Regression_Ex()

    c.backTesting()
#    c.predictNextDayPrice()

    print ("Done !!")