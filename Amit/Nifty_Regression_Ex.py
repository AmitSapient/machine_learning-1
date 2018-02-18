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
import StockPredictionHelper
from sklearn.neural_network import MLPRegressor

class Nifty_Regression_Ex:

    def __init__(self):
        self.helper = StockPredictionHelper.StockPredictionHelper()
        
    def addOIAndFOMoreFeatures(self,df):
        
        days = 5
        df = self.helper.addPrevDaysAsFeature(df,'call_contracts',days)
        df = self.helper.addPrevDaysAsFeature(df,'put_contracts',days)
#        df = self.helper.addPrevDaysAsFeature(df,'call_open_int',days)
#        df = self.helper.addPrevDaysAsFeature(df,'put_open_int',days)
        df = df.drop(['call_open_int','put_open_int'],1)
        days = 5
        df = self.helper.addChangeFromPreviousDay(df,'call_contracts',days)
        df = self.helper.addChangeFromPreviousDay(df,'put_contracts',days)
#        df = self.helper.addChangeFromPreviousDay(df,'call_open_int',days)
#        df = self.helper.addChangeFromPreviousDay(df,'put_open_int',days)
        
        return df

    def addAllFeaturesOtherThanOI(self,df, flag):
        
        df = self.helper.addNiftyCloseAsFeature(df, flag)
        df['nifty_close'] = df['nifty_close'].fillna((df['nifty_close'].mean()))

#        df = df.fillna( df.mean()) 
        
        df = self.helper.moving_average(df)
        
#        df = self.helper.MACD(df)
#        df = self.helper.calculateRSI(df) 
        
        df = self.helper.addMACDAsFeature(df)       

        df = self.helper.addPrevDaysAsFeature(df,'close',60)
        
        return df        
        
        
    def backTesting(self, ticker, flag):
        print("\n\n\n\n\n***********  New Run of Nifty_Regression_Ex **********************\n\n")
        pd.set_option('display.expand_frame_repr', False)
#        np.set_printoptions(formatter={'float_kind':'{:f}'.format})
        np.set_printoptions(suppress=True)
        
        df = self.helper.getData(ticker, flag)
#        df = self.helper.getDataFromNSE(ticker, flag)
        
        df = df.fillna(df.mean())
        # Add another column next_day_close exct duplicate of close. Later we shift to make prev close.
        # First shift 'close' by -1 because today's close depends on all the feature values of yesterday. To predict today's close
        # we need the open, high, lo, volumne of yesterday because we dont have these values yet for today ! 
        df['next_day_close'] = df['close'].shift(-1)
        
        #add change & change % from previous day
        df = self.helper.addChangeFromPreviousDay(df,'close',1)
        
        #old fn for total OI (options+FO together)
#        df = self.helper.addOpenInterestData(df, ticker)
        #New OI function
#        df = self.helper.addOptionsANDFoData(df, ticker)
#        df = self.addOIAndFOMoreFeatures(df)
        
        df = self.addAllFeaturesOtherThanOI(df, flag)
        
        df = df.dropna(axis=0, how='any')
        Xdf_Tomorrow = df.tail(1)  
         
        
#        print('df after all NA drop - \n', df)
        print('\n\n Final df columns - \n', df.columns)
        
        Xdf_Tomorrow = Xdf_Tomorrow.drop(['next_day_close'],1)
        filename = 'datasets\\'+ticker+'_OI.csv'
        df.to_csv(filename)

#        print ("DF Columns - ",df.columns)
       
        X_Tomorrow = np.array(Xdf_Tomorrow)
        X = np.array(df.drop(['next_day_close'],1))
        X = preprocessing.scale(X)
#        X = preprocessing.normalize(X)
        
        y = np.array(df['next_day_close'])
#        y = np.array(df['close'])
#        print ("Y values - ", y[-5:])

#        print("Dataset lenghts - ",(len(X), len(y)))


        #X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)
        #train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
        X_train, X_rest, y_train, y_rest = model_selection.train_test_split(X, y, test_size=0.4, random_state=42)
        X_test, X_Validation, y_test, y_Validation = model_selection.train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)
        
#        print("Dataset lenghts Train set- ",(len(X_train), len(y_train)))
#        print("Dataset lenghts Test set- ",(len(X_test), len(y_test)))
#        print("Dataset lenghts validation set- ",(len(X_Validation), len(y_Validation)))
        
        #result_df = pd.DataFrame()
#        result_df.columns = [['Actual Close', 'Linear Reg', 'SVM', 'Decision Tree']]
#        self.result_df.loc[0, 'Actual Close'] = X_Validation[0]

#        df.loc[df.shape[0]] = ['d', 3] 
        
#        self.callRandomForest(X_train, X_test, y_train, y_test,X_Validation , y_Validation, X_Tomorrow)
#        self.callDecisionTreeRegressor(X_train, X_test, y_train, y_test,X_Validation , y_Validation, X_Tomorrow)
#        self.callGradientBoosting(X_train, X_test, y_train, y_test,X_Validation , y_Validation, X_Tomorrow)
#        
#        self.callLinearRegression(X_train, X_test, y_train, y_test,X_Validation, y_Validation, X_Tomorrow )
##        self.callLogisticRegression(X_train, X_test, y_train, y_test,X_Validation , y_Validation , X_Tomorrow)
#        self.callSVM(X_train, X_test, y_train, y_test,X_Validation , y_Validation, X_Tomorrow)
#        self.callXGBClassifier(X_train, X_test, y_train, y_test,X_Validation , y_Validation, X_Tomorrow)
#        self.callKNN(X_train, X_test, y_train, y_test,X_Validation , y_Validation, X_Tomorrow)
#    
        #Neural Network
        self.callNeuralNetwork(X_train, X_test, y_train, y_test,X_Validation , y_Validation, X_Tomorrow)

    def callNeuralNetwork(self, X_train, X_test, y_train, y_test,X_Validation , y_Validation, X_Tomorrow ):
        #Random forest
#        clf = MLPRegressor(alpha=0.001, solver='lbfgs', hidden_layer_sizes = (100,), max_iter = 100000, 
#                 activation = 'logistic', learning_rate = 'adaptive')
        clf = MLPRegressor(alpha=0.001, hidden_layer_sizes = (10,), max_iter = 100000, 
                 activation = 'logistic', verbose = 'True', learning_rate = 'adaptive')

        clf.fit(X_train, y_train)
        
        y_predicted = clf.predict(X_Validation)
        accuracy = clf.score(X_test, y_test)
        diff = y_Validation - y_predicted
        diffArray = np.column_stack((y_Validation , y_predicted, diff))
        print("\n NeuralNetwork diffArray  - \n", diffArray)
        print ("\n****** NeuralNetwork Accuracy  - ",accuracy)

        mse = mean_squared_error(y_Validation, y_predicted)
        print ("Mean Squred Error", mse)
        variance = np.var(y_predicted)         
        print ("Varinace - ", variance)
        
        y_Tomorrow = clf.predict(X_Tomorrow)
        print ("X_Tomorrow close - ", X_Tomorrow[0][0])
        print ("y_Tomorrow - ", y_Tomorrow)
        
    def callRandomForest(self, X_train, X_test, y_train, y_test,X_Validation , y_Validation, X_Tomorrow ):
        #Random forest
        clf = RandomForestRegressor(min_samples_leaf=1, max_features=0.5, n_estimators=20, random_state=42, oob_score=True)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        y_predicted = clf.predict(X_Validation)
        diff = y_Validation - y_predicted
        diffArray = np.column_stack((y_Validation , y_predicted, diff))
#        print("\n diffArray  - \n", diffArray)
        print ("\n****** RandomForestClassifier Accuracy  - ",accuracy)

        mse = mean_squared_error(y_Validation, y_predicted)
        print ("Mean Squred Error", mse)
        variance = np.var(y_predicted)         
        print ("Varinace - ", variance)
        
        y_Tomorrow = clf.predict(X_Tomorrow)
        print ("X_Tomorrow close - ", X_Tomorrow[0][0])
        print ("y_Tomorrow - ", y_Tomorrow)

    def callDecisionTreeRegressor(self, X_train, X_test, y_train, y_test,X_Validation , y_Validation, X_Tomorrow):
        #Decision Tree
        clf = tree.DecisionTreeRegressor()
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        
        y_predicted = clf.predict(X_Validation)
        diff = y_Validation - y_predicted
        diffArray = np.column_stack((y_Validation , y_predicted, diff))
#        print("\n diffArray  - \n", diffArray)
        print ("\n****** Decision Tree Accuracy  - ",accuracy)  
        mse = mean_squared_error(y_Validation, y_predicted)
        print ("Mean Squred Error", mse)
        variance = np.var(y_predicted)         
        print ("Varinace - ", variance)
        
        y_Tomorrow = clf.predict(X_Tomorrow)
        print ("X_Tomorrow - ", X_Tomorrow[0][0])
        print ("y_Tomorrow - ", y_Tomorrow)

        
    def callGradientBoosting(self, X_train, X_test, y_train, y_test,X_Validation , y_Validation, X_Tomorrow):
        #GradientBoostingClassifier
        clf= GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        y_predicted = clf.predict(X_Validation)
        diff = y_Validation - y_predicted
        diffArray = np.column_stack((y_Validation , y_predicted, diff))
#        print("\n diffArray  - \n", diffArray)
        print ("\n****** callGradientBoosting Accuracy  - ",accuracy)

        mse = mean_squared_error(y_Validation, y_predicted)
        print ("Mean Squred Error", mse)
        variance = np.var(y_predicted)         
        print ("Varinace - ", variance)        

        y_Tomorrow = clf.predict(X_Tomorrow)
        print ("X_Tomorrow - ", X_Tomorrow[0][0])
        print ("y_Tomorrow - ", y_Tomorrow)

    def callLinearRegression(self, X_train, X_test, y_train, y_test,X_Validation , y_Validation, X_Tomorrow):
        # Linear regression
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        
        y_predicted = clf.predict(X_Validation)
#        accuracy = clf.score(X_Validation, y_Validation )
#        print ("\n\n****** LinearRegression Accuracy  - ",accuracy)
        
        diff = y_Validation - y_predicted
        diffArray = np.column_stack((y_Validation , y_predicted, diff))
#        print("\n diffArray  - \n", diffArray)
        print ("\n****** LinearRegression Accuracy  - ",accuracy)

        mse = mean_squared_error(y_Validation, y_predicted)
        print ("Mean Squred Error", mse)
        variance = np.var(y_predicted)         
        print ("Varinace - ", variance)

        y_Tomorrow = clf.predict(X_Tomorrow)
        print ("X_Tomorrow close - ", X_Tomorrow[0][0])
        print ("y_Tomorrow - ", y_Tomorrow)
        

    def callKNN(self, X_train, X_test, y_train, y_test,X_Validation , y_Validation, X_Tomorrow):
         #KNN
        clf = KNeighborsRegressor()
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        y_predicted = clf.predict(X_Validation)
        diff = y_Validation - y_predicted
        diffArray = np.column_stack((y_Validation , y_predicted, diff))
#        print("\n diffArray  - \n", diffArray)
        print ("\n****** KNeighborsRegressor Accuracy  - ",accuracy)

        mse = mean_squared_error(y_Validation, y_predicted)
        print ("Mean Squred Error", mse)
        variance = np.var(y_predicted)         
        print ("Varinace - ", variance)        
        
        y_Tomorrow = clf.predict(X_Tomorrow)
        print ("X_Tomorrow close - ", X_Tomorrow[0][0])
        print ("y_Tomorrow - ", y_Tomorrow)
        
    def callXGBClassifier(self, X_train, X_test, y_train, y_test,X_Validation , y_Validation, X_Tomorrow):
        #XGBClassifier
        clf = XGBRegressor()
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        y_predicted = clf.predict(X_Validation)
        diff = y_Validation - y_predicted
        diffArray = np.column_stack((y_Validation , y_predicted, diff))
#        print("\n diffArray  - \n", diffArray)
        print ("\n****** XGBRegressor Accuracy  - ",accuracy)

        mse = mean_squared_error(y_Validation, y_predicted)
        print ("Mean Squred Error", mse)
        variance = np.var(y_predicted)         
        print ("Varinace - ", variance)        
        
        y_Tomorrow = clf.predict(X_Tomorrow)
        print ("X_Tomorrow close - ", X_Tomorrow[0][0])
        print ("y_Tomorrow - ", y_Tomorrow)

    def callLogisticRegression(self, X_train, X_test, y_train, y_test,X_Validation , y_Validation , X_Tomorrow):
        # Linear regression
        clf = LogisticRegression(penalty='l2',C=.01)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        y_predicted = clf.predict(X_Validation)
        diff = y_Validation - y_predicted
        diffArray = np.column_stack((y_Validation , y_predicted, diff))
#        print("\n diffArray  - \n", diffArray)
        print ("\n****** LogisticRegression Accuracy  - ",accuracy)

        mse = mean_squared_error(y_Validation, y_predicted)
        print ("Mean Squred Error", mse)
        variance = np.var(y_predicted)         
        print ("Varinace - ", variance)        

        y_Tomorrow = clf.predict(X_Tomorrow)
        print ("X_Tomorrow close - ", X_Tomorrow[0][0])
        print ("y_Tomorrow - ", y_Tomorrow)
        
    def callSVM(self, X_train, X_test, y_train, y_test,X_Validation , y_Validation, X_Tomorrow ):
        # SVM        
        clf = svm.SVR(kernel='poly')
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        y_predicted = clf.predict(X_Validation)
        diff = y_Validation - y_predicted
        diffArray = np.column_stack((y_Validation , y_predicted, diff))
#        print("\n diffArray  - \n", diffArray)
        print ("\n****** SVM Accuracy  - ",accuracy)

        mse = mean_squared_error(y_Validation, y_predicted)
        print ("Mean Squred Error", mse)
        variance = np.var(y_predicted)         
        print ("Varinace - ", variance)        

        y_Tomorrow = clf.predict(X_Tomorrow)
        print ("X_Tomorrow close - ", X_Tomorrow[0][0])
        print ("y_Tomorrow - ", y_Tomorrow)
        

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
    thisObj = Nifty_Regression_Ex()
    #MARUTI, AUROPHARMA, 8KMILES, FEDERALBNK, NCC, GMRINFRA, HDFCBANK
    ticker = 'HDFCBANK'
    thisObj.backTesting(ticker, 'real_time')
#    thisObj.backTesting(ticker, 'csv')
    
#    thisObj.predictNextDayPrice()

    print ("Done !!")