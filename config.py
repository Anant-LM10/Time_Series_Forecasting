import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
from sklearn.linear_model import LinearRegression, Lasso
from statsmodels.nonparametric.smoothers_lowess import lowess

import optuna
import pymannkendall as mk
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import MSTL,STL

sales = pd.read_csv(r"store-sales-time-series-forecasting\train.csv")
test_sales = pd.read_csv(r"store-sales-time-series-forecasting\test.csv")
oil = pd.read_csv(r"store-sales-time-series-forecasting\oil.csv")
holidays = pd.read_csv(r"store-sales-time-series-forecasting\holidays_events.csv")
stores = pd.read_csv(r"store-sales-time-series-forecasting\stores.csv")




classif = pd.read_csv(r"classif.csv")
sales.drop(["id"],axis=1,inplace=True)

sales["date"] = sales["date"].apply(lambda x: datetime.strptime(x,"%Y-%m-%d"))

sales["tsid"] = sales["family"] +"_"+ sales["store_nbr"].astype(int).astype(str)

sales.set_index("date",inplace=True)

for i in ["2013-12-25","2014-12-25","2015-12-25","2016-12-25"]:
    sales.loc[i,"sales"] = 0
    sales.loc[i,"onpromotion"] = 0

sales = sales.sort_index()

sales.reset_index().merge(stores[["store_nbr","city"]], left_on="store_nbr", right_on="store_nbr").set_index("date")

holidays["date"] = pd.to_datetime(holidays["date"])
holidays.set_index(["date"], inplace=True)
holidays = holidays["2013-01-01":]
holidays["Day_of_Week"] = holidays.index.day_of_week
#holidays_1 = holidays[holidays["locale_name"].isin(["Ecuador","Quito"])]

tmp_2 = sales.join(holidays.drop(["Day_of_Week"],axis=1)).fillna("NA")

tmp_2["Day_of_Week"] = tmp_2.index.day_of_week

holidays_join = holidays.copy(deep=True)
holidays_join["Holiday"] = 0
holidays_join["Additional"] = np.where(holidays_join["type"] == "Additional", 1, 0)
holidays_join["Christmas"] = np.where((holidays_join.index.month == 12) & holidays_join.index.day.isin([21,22,23,24]), 1, 0)

holidays_join["Holiday"] = np.where(((holidays_join["type"].isin(["Transfer", "Bridge", "Holiday"])) & (holidays_join["transferred"] == False)), 1, 0)
holidays_join.loc["2016-04-17":"2016-04-23","Additional"] = 1

for i in range(5):
    holidays_join.loc[str(2013+i)+"-12-21":str(2013+i)+"-12-24","Additional"] = 0
    holidays_join.loc[str(2013+i)+"-12-26","Holiday"] = 1

holidays_join.loc[(holidays_join.index.month == 12) & (holidays_join.index.day == 26), "Additional"] = 0

oil["dcoilwtico_in"] = oil["dcoilwtico"].interpolate()
oil["date"] = oil["date"].apply(lambda x: datetime.strptime(x,"%Y-%m-%d"))
oil = oil.set_index("date")
oil = oil.fillna(method="bfill")
oil.drop("dcoilwtico",axis=1,inplace=True)

class Segmentation:

    def __init__(self, data):
        self.data = data

    def cov(self):
        return (np.std(self.data["sales"])/np.mean(self.data["sales"])) ** 2

    def volume(self):
        df = self.data[self.data["sales"] > 0].reset_index()
        return ((df["date"] - df["date"].shift(1)).astype(str).str.replace(' days','').replace('NaT', np.nan)).dropna().astype(int).mean()

    def segmentation(self):
        segment = np.where(((self.volume() < 1.32) & (self.cov() < 0.49)), "Smooth", np.where((self.volume() < 1.32) & (self.cov() >= 0.49), "Erratic", np.where((self.volume() >= 1.32) & (self.cov() < 0.49), "Intermittent", "Lumpy")))
        return segment

    def trend(self):
        return mk.original_test(list(self.data["sales"])).trend


def detect_outliers(data, freq):
    outliers = []
    seasonal_iqrs = seasonal_iqr(data, freq)

    for i in range(1, freq + 1):
        seasonal_data = data[data["Day_of_Year"] == i]['sales']
        lower_bound, upper_bound = seasonal_iqrs[i - 1][1], seasonal_iqrs[i - 1][2]
        seasonal_outliers = seasonal_data[(seasonal_data < lower_bound) | (seasonal_data > upper_bound)]
        outliers.extend(seasonal_outliers.index)
    return outliers

class preprocessing():

    def __init__(self, data, tsid, trend):
        self.data = data
        self.tsid = tsid
        self.trend = trend
        
    def sales_data(self):
        
        print("Sales Data Pre-Processing started for ",self.tsid)
        sales1 = self.data[(self.data["family"] == self.tsid.split("_")[0]) & (self.data["store_nbr"] == int(self.tsid.split("_")[1]))]
        for i in ["2013-12-25","2014-12-25","2015-12-25","2016-12-25"]:
            sales1.loc[i,"sales"] = 0
            sales1.loc[i,"onpromotion"] = 0
            sales1.loc[i,"family"] = self.tsid.split("_")[0]
            sales1.loc[i,"store_nbr"] = int(self.tsid.split("_")[1])
            sales1.loc[i,"city"] = list(stores[stores["store_nbr"] == int(self.tsid.split("_")[1])]["city"])[0]

        sales1["Country"] = "Ecuador"
        sales1["Day_of_Week"] = sales1.index.day_of_week
        sales1["Week_of_Year"] = sales1.index.isocalendar().week
        sales1["Month_of_Year"] = sales1.index.month
        sales1 = sales1.join(oil).sort_index().fillna(method="bfill")

        return sales1

    def holiday_data(self):
        
        sales = self.sales_data()
        print(sales.shape)
        print("Holiday Pre-Processing started for ",self.tsid)
        holidays_GI1 = holidays_join[holidays_join["locale_name"].isin(["Ecuador",list(stores[stores["store_nbr"] == int(self.tsid.split("_")[1])]["city"])[0]])]
        holidays_GI1 = holidays_GI1.reset_index()[["date","locale_name", "Holiday", "Additional", "Christmas"]]

        holidays_GI1.dropna(inplace=True)

        holiday_add = pd.concat((sales[sales["Day_of_Week"] == 6]["Day_of_Week"], holidays_GI1.set_index(["date"]))).sort_index().drop(["locale_name"],axis=1)
        holiday_add.fillna(0, inplace=True)

        holiday_add.columns = ["Sunday","Holiday","Additional","Christmas"]
        holiday_add["Sunday"] = holiday_add["Sunday"]/6

        holiday_add = holiday_add[(holiday_add["Additional"] + holiday_add["Holiday"] + holiday_add["Christmas"] + holiday_add["Sunday"]) >= 1]
        holiday_add.reset_index(inplace=True)
        holiday_add["Days_Till_Next_Holiday"] = (holiday_add["date"] - holiday_add["date"].shift(1)).dt.days

        idx = list(holiday_add[holiday_add["date"].duplicated(False)][holiday_add[holiday_add["date"].duplicated(False)]["Sunday"] == 1].index)
        holiday_add.drop(idx, inplace=True)
        holiday_add.dropna(inplace=True)

        holiday_add = pd.concat((holiday_add, pd.DataFrame(holiday_add[holiday_add["Days_Till_Next_Holiday"] == 2]["date"] - timedelta(days=1), columns=["date"]).reset_index(drop=True)))
        holiday_add.sort_values("date", ascending=True,inplace=True)

        holiday_add["Additional"].fillna(1, inplace=True)
        holiday_add["Holiday"].fillna(0, inplace=True)
        holiday_add["Christmas"].fillna(0, inplace=True)
        holiday_add.reset_index(inplace=True, drop=True)

        holiday_add["Sunday"] = np.where(holiday_add["date"].dt.day_of_week == 6, 1, 0)
        holiday_add["Days_Till_Next_Holiday"] = (holiday_add["date"] - holiday_add["date"].shift(1)).dt.days
        holiday_add.set_index("date", drop=True, inplace=True)

        return holiday_add

    def final_data(self):
        holiday_add = self.holiday_data()
        print("Holiday Pre-Processing ended for ",self.tsid)
        sales = self.sales_data()
        print("Sales Data Pre-Processing ended for ",self.tsid)
        sales = sales.join(holiday_add[["Holiday","Additional","Christmas"]])
        sales.fillna(0, inplace=True)
        for i in range(0,4):
            sales.loc[datetime.strptime((str(2013+i)+"-12-26"), "%Y-%m-%d"),"Holiday"] = 1.0
            sales.loc[datetime.strptime((str(2013+i)+"-12-26"), "%Y-%m-%d"),"Additional"] = 0.0
        sales = sales.drop(["store_nbr","family","city","Country"], axis=1)
        return sales

    def detrending(self):
        
        sales = self.final_data()
        lw = []
        if self.trend != "no trend":    
            print(sales.head())
            X = list(range(0,len(sales)))
            y = sales["sales"]
            lw = lowess(y, X, frac=0.05)
            sales["sales"] = sales["sales"] - pd.DataFrame(lw, index=sales.index, columns=["id","trend"])["trend"]
            lw = pd.DataFrame(lw, index=sales.index, columns=["id","trend"])["trend"]
        return sales, lw

def expanding_wt_mean(x, al):
    
    weight = np.ones(len(x))
    for i in range(len(x)):
        weight[i] = np.power((1-al), len(x) - 1 - i)
        
    return np.sum(weight * x) / np.sum(weight)


def rolling_wt_mean(x,al):
    
    weight = np.ones(len(x))
    for i in range(7):
        weight[i] = np.power((1-al), 6 - i)
    return np.sum(weight * x) / np.sum(weight)

class Forecasting:
    
    def __init__(self, model, params, init_data, start_date, end_date, drop=[]):
        self.model = model
        self.params = params
        self.init_data = init_data
        self.start_date = start_date
        self.end_date = end_date
        self.drop = drop

    def lagged(self,X):
        X["sales_lag1"] = X["sales"].shift(1).fillna(method='bfill')
        X["sales_lag7"] = X["sales"].shift(7).fillna(method='bfill')
        X["sales_lag363"] = X["sales"].shift(365).fillna(method='bfill')
        return X

    def creating_dataset(self, N = None):
        X = self.init_data[:datetime.strptime(self.start_date, "%Y-%m-%d")]
        X_1 = self.init_data[datetime.strptime(self.start_date, "%Y-%m-%d"):datetime.strptime(self.end_date, "%Y-%m-%d")]
        X = self.lagged(X)
        X["sales_expanding_wt"] = X["sales"].expanding().agg(expanding_wt_mean,0.2).shift(1).fillna(method='bfill')
        X["sales_wt_mean7"] = X["sales"].rolling(7).agg(rolling_wt_mean,0.6).shift(1).fillna(method='bfill')
        X = X.drop(self.drop, axis=1)
        return X, X_1

    def construct_train(self,predicted, index, X1, X_1, alp, N=None):
        X1.loc[index,"sales"] = predicted
        next = X1.index.max() + timedelta(days=1)
        X1.loc[next,"onpromotion"] = X_1.loc[next,"onpromotion"]
        X1.loc[next,"Holiday"] = X_1.loc[next,"Holiday"]
        X1.loc[next,"Additional"] = X_1.loc[next,"Additional"]
        X1.loc[next,"Christmas"] = X_1.loc[next,"Christmas"]
        X1.loc[next,"dcoilwtico_in"] = X_1.loc[next,"dcoilwtico_in"]
        if any("Day_of_Week" in x for x in list(X1.columns)):
            X1.loc[next,"Day_of_Week"] = X_1.loc[next,"Day_of_Week"]
        if any("Week_of_Year" in x for x in list(X1.columns)):
            X1.loc[next,"Week_of_Year"] = X_1.loc[next,"Week_of_Year"]
        if any("Month_of_Year" in x for x in list(X1.columns)):
            X1.loc[next,"Month_of_Year"] = X_1.loc[next,"Month_of_Year"]
        X1 = self.lagged(X1)
        if alp == 999:
            alpha = SimpleExpSmoothing(X1["sales"]).fit(optimized=True).params["smoothing_level"]
        else : alpha = alp
        X1["sales_expanding_wt"] = X1["sales"].expanding().agg(expanding_wt_mean, alpha).shift(1).fillna(method='bfill')
        X1["sales_wt_mean7"] = X1["sales"].rolling(7).agg(rolling_wt_mean, (alpha)).shift(1).fillna(method='bfill')
        X1.drop(self.drop,axis=1,inplace=True)

        return X1

    def fit(self,X, y = None, drop = []):
        
        last_day = X.index.max() - timedelta(days=1)
        X_train = X.drop(["sales"],axis=1)
        X_train = pd.DataFrame(X_train, columns=X.drop(["sales"],axis=1).columns, index=X.index)
        ls = self.model()
        ls.set_params(**self.params)
        ls_fit = ls.fit(X_train.loc[:last_day], X.loc[:last_day,"sales"])
        return ls_fit

    def predict(self,X, N=None):
        X, X_1 = self.creating_dataset(N)
        pred1 = []
        
        for i in range(len(X_1[:datetime.strptime(self.end_date,"%Y-%m-%d")])):
            next_day = X_1.index.min() + timedelta(days=i)
            ls_fit = self.fit(X[:X_1.index.min() + timedelta(days=i-1)])
            X_train = pd.DataFrame(X.drop(["sales"],axis=1), columns=X.drop(["sales"],axis=1).columns, index=X.index)
            pred = ls_fit.predict(X_train[next_day:next_day])
            pred1.append(pred)
            if next_day < datetime.strptime(self.end_date,"%Y-%m-%d"):
                X = self.construct_train(pred,next_day, X, X_1, alp=0.2, N = N)
            else: break
        return np.array(pred1)
    
class Stat_Models():
    
    def __init__(self, model, init_data, tsid, start, end, params, trend, backtesting=True):
        self.model = model
        self.init_data = init_data
        self.tsid = tsid
        self.start = start
        self.end = end
        self.params = params
        self.backtesting = backtesting
        self.trend = trend
        
    def prophet_concurrent(self):

        df = self.init_data[self.init_data["tsid"] == self.tsid]
        
        train = df[:datetime.strptime(self.start, "%Y-%m-%d")]["sales"].reset_index()
        train.columns = ["ds","y"]

        if self.backtesting:
        
            test = df[datetime.strptime(self.start, "%Y-%m-%d"):datetime.strptime(self.end, "%Y-%m-%d")]["sales"].reset_index()
            test.columns = ["ds","y"]

        holidays = holidays_join[holidays_join["locale_name"].isin(list([stores[stores["store_nbr"] == int(self.tsid.split("_")[1])]["city"].values[0], stores[stores["store_nbr"] == int(self.tsid.split("_")[1])]["state"].values[0], "Ecuador"]))].reset_index()
        holidays["Holiday"] = holidays["Holiday"] + holidays["Additional"] + holidays["Christmas"]
        holidays = holidays[holidays["Holiday"] == 1]

        holidays = holidays[["date","description"]]
        holidays.rename(columns={"date":"ds","description":"holiday"},inplace=True)

        day = (datetime.strptime(self.end,"%Y-%m-%d") - datetime.strptime(self.start,"%Y-%m-%d")).days + 1
        
        pr = Prophet(**self.params)
        pr_fit = pr.fit(train)
        predict = pr.make_future_dataframe(periods=day, include_history=False)

        X1 = pr.predict(predict)
        print("Forecasting complete for tsid: ",self.tsid)

        return X1.set_index("ds")["yhat"]

    def exponential_smoothing(self):
        print("Pre-processing started for:", self.tsid)
        pre = preprocessing(self.init_data,self.tsid,self.trend)
        sales_tsid = pre.final_data()
        print("Pre-processing completed for:", self.tsid)
        sd = MSTL(sales_tsid["sales"], periods=[7,365])
        sd_fit = sd.fit()
        print("Decomposition completed for:", self.tsid)
        #Seasonal_365
        seas_365 = pd.DataFrame(sd_fit.seasonal["seasonal_365"])
        seas_365.columns = ["sales"]

        train = seas_365[:datetime.strptime(self.start,"%Y-%m-%d")]
        
        if self.backtesting:
            test = seas_365[datetime.strptime(self.start,"%Y-%m-%d"):datetime.strptime(self.end,"%Y-%m-%d")]
        
        tes = ExponentialSmoothing(train, seasonal="add", seasonal_periods=365)
        tes_fit = tes.fit(**self.params[0])
        
        day = (datetime.strptime(self.end,"%Y-%m-%d") - datetime.strptime(self.start,"%Y-%m-%d")).days + 1
        
        tes_365 = tes_fit.forecast(day)
        
        #Seasonal_7
        seas_7 = pd.DataFrame(sd_fit.seasonal["seasonal_7"])
        seas_7.columns = ["sales"]

        train = seas_7[["sales"]][:datetime.strptime(self.start,"%Y-%m-%d")]
        
        if self.backtesting:
            test = seas_7[["sales"]][datetime.strptime(self.start,"%Y-%m-%d"):datetime.strptime(self.end,"%Y-%m-%d")]

        tes = ExponentialSmoothing(train, seasonal="add", seasonal_periods=7)
        tes_fit = tes.fit(**self.params[1])

        tes_7 = tes_fit.forecast(day)
        
        #Trend
        trend = pd.DataFrame(sd_fit.trend.dropna())
        trend.columns = ["sales"]
        trend = trend.merge(sales_tsid.drop(["sales"], axis=1), left_index=True, right_index=True)
        
        train = trend[["sales"]][datetime.strptime(self.start,"%Y-%m-%d") - timedelta(days=365):datetime.strptime(self.start,"%Y-%m-%d")]
        
        if self.backtesting:    
            test = trend[["sales"]][datetime.strptime(self.start,"%Y-%m-%d"):datetime.strptime(self.end,"%Y-%m-%d")]

        tes = ExponentialSmoothing(train, trend = self.params[2]["trend"], damped_trend=self.params[2]["damped_trend"])
        tes_fit = tes.fit(smoothing_level=self.params[2]["smoothing_level"], smoothing_trend=self.params[2]["smoothing_trend"])

        tes_2 = tes_fit.forecast(day)
        return np.array(tes_365) + np.array(tes_7) + np.array(tes_2)
    
class Tuning():
    
    def __init__(self, model, init_data, tsid, start, end, col, trial, check_trend, period, drop=[],scaling=False):
        self.model = model
        self.scaling = scaling
        self.init_data = init_data
        self.start = start
        self.end = end
        self.drop = drop
        self.trial = trial
        self.col = col
        self.tsid = tsid
        self.check_trend = check_trend
        self.period = period


    def rf_optimization(self, trail):
        params = {"max_depth": trail.suggest_int("max_depth",7,59,step=2), "n_estimators":trail.suggest_int("n_estimators",250,1000,step=250),"max_features":trail.suggest_categorical("max_features",["log2","sqrt"]),"min_samples_split":trail.suggest_int("min_samples_split",2,10,step=2),"random_state":42, "n_jobs":-1}
        return params

    def lgb_optimization(self, trail):
        params = {"max_depth": trail.suggest_int("max_depth",7,59,step=2), "learning_rate":trail.suggest_float("learning_rate",0.1,0.8,step=0.1),"bagging_fraction":trail.suggest_float("bagging_fraction",0.8,1,step=0.1),"feature_fraction":trail.suggest_float("feature_fraction",0.8,1,step=0.1), "reg_alpha":trail.suggest_categorical("reg_alpha",[0.1, 0.5, 1, 10]),"verbosity":-1, "random_state":42,"n_jobs":-1}
        return params

    def xgb_optimization(self, trail):
        params = {"max_depth": trail.suggest_int("max_depth",7,69,step=2), "eta":trail.suggest_float("eta",0.1,0.8,step=0.1),"lambda":trail.suggest_categorical("lambda",[0.1, 0.5, 1, 10]),"booster":trail.suggest_categorical("booster",["gbtree"]),"n_jobs":-1, "n_estimators":trail.suggest_categorical("n_estimators",[100, 200, 250, 500]), "subsample":trail.suggest_float("subsample",0.7,1, step=0.1)}
        return params
    
    def prophet_optimization(self, trial):
        params = {"n_changepoints" : trial.suggest_int("n_changepoints",50,250,step=50), 
                  "changepoint_prior_scale" : trial.suggest_float("changepoint_prior_scale",0.5,1.1,step=0.1),
                  "holidays_prior_scale" : trial.suggest_categorical("holidays_prior_scale",[10,25,35,50]),
                  "seasonality_prior_scale" : trial.suggest_categorical("seasonality_prior_scale",[5,10,20]), 
                  "seasonality_mode" : trial.suggest_categorical("seasonality_mode",["additive","multiplicative"]),
                  "changepoint_range" : trial.suggest_float("changepoint_range",0.6,0.95,step=0.05)}
        return params
    
    def ExpSmoothing_Fit_Optimization(trial):
        params={"smoothing_level":trial.suggest_float("smoothing_level",0,0.1,step=0.05), "smoothing_seasonal":trial.suggest_float("smoothing_seasonal",0.7,1,step=0.05),"smoothing_trend":trial.suggest_float("smoothing_trend",0.7,1,step=0.05)}
        return params

    def ExpSmoothing_trend_optimization(trial):
        params1 = {"trend":trial.suggest_categorical("trend",["add"]), "damped_trend":trial.suggest_categorical("damped_trend",[True, False])}
        return params1

    def objective(self, trial: optuna.Trial):
        
        if self.model in [RandomForestRegressor, XGBRegressor, LGBMRegressor]:
            df = self.init_data[self.init_data["tsid"] == self.tsid]
            df.drop("tsid",inplace=True,axis=1)
            if self.model == RandomForestRegressor:
                fcst = Forecasting(RandomForestRegressor,self.rf_optimization(trial),df,self.start,self.end,drop=self.drop)
            elif self.model == XGBRegressor:
                fcst = Forecasting(XGBRegressor,self.xgb_optimization(trial),df,self.start,self.end,drop=self.drop)
            elif self.model == LGBMRegressor:
                fcst = Forecasting(LGBMRegressor,self.lgb_optimization(trial),df,self.start,self.end,drop=self.drop)
            X, X1 = fcst.creating_dataset()
            ls_fit = fcst.fit(X)
            predict = fcst.predict(X)
            
            mae = np.mean(np.abs(X1[datetime.strptime(self.start,"%Y-%m-%d"):datetime.strptime(self.end,"%Y-%m-%d")][self.col] - pd.DataFrame(predict,index=X1[:datetime.strptime(self.end,"%Y-%m-%d")].index, columns=[self.col]).fillna(0)[self.col]))    
        
        elif self.model == Prophet:
            df = self.init_data[self.init_data["tsid"] == self.tsid]
            train = df[:datetime.strptime(self.start, "%Y-%m-%d") - timedelta(days=1)]["sales"].reset_index()
            train.columns = ["ds","y"]

            test = df[datetime.strptime(self.start, "%Y-%m-%d"):datetime.strptime(self.end, "%Y-%m-%d")]["sales"].reset_index()
            test.columns = ["ds","y"]
            
            
            holidays = holidays_join[holidays_join["locale_name"].isin(list([stores[stores["store_nbr"] == int(self.tsid.split("_")[1])]["city"].values[0], stores[stores["store_nbr"] == int(self.tsid.split("_")[1])]["state"].values[0], "Ecuador"]))].reset_index()
            holidays["Holiday"] = holidays["Holiday"] + holidays["Additional"] + holidays["Christmas"]
            holidays = holidays[holidays["Holiday"] == 1]

            holidays = holidays[["date","description"]]
            holidays.rename(columns={"date":"ds","description":"holiday"},inplace=True)            
            params = self.prophet_optimization(trial)

            pr = Prophet(n_changepoints = params["n_changepoints"], 
                         changepoint_prior_scale = params["changepoint_prior_scale"], 
                         holidays_prior_scale = params["holidays_prior_scale"], 
                         seasonality_prior_scale = params["seasonality_prior_scale"],
                         seasonality_mode = params["seasonality_mode"],
                         changepoint_range = params["changepoint_range"], 
                         growth = "linear", yearly_seasonality=True, weekly_seasonality=True, holidays=holidays)
            
            pr_fit = pr.fit(train)
            day = (datetime.strptime(self.end,"%Y-%m-%d") - datetime.strptime(self.start,"%Y-%m-%d")).days + 1
            
            pred = pr.make_future_dataframe(periods=day, include_history=False)
            predict = pr.predict(pred)
            mae = np.mean(np.abs(test.set_index("ds")["y"] - predict.set_index("ds")["yhat"]))    

        else :    
            
            if self.check_trend == False:
                train = self.init_data[:datetime.strptime(self.start, "%Y-%m-%d")]
                test = self.init_data[datetime.strptime(self.start, "%Y-%m-%d"):datetime.strptime(self.end, "%Y-%m-%d")]
                if self.period == 365:
                    exp = ExponentialSmoothing(train, seasonal="add", seasonal_periods=365)
                    exp_fit = exp.fit(**ExpSmoothing_Fit_Optimization(trial))
                elif self.period == 7:
                    exp = ExponentialSmoothing(train, seasonal="add", seasonal_periods=7)
                    exp_fit = exp.fit(**ExpSmoothing_Fit_Optimization(trial))            
            else: 
                train = self.init_data[datetime.strptime(self.start, "%Y-%m-%d") - timedelta(days=365):datetime.strptime(self.start, "%Y-%m-%d")]
                test = self.init_data[datetime.strptime(self.start, "%Y-%m-%d"):datetime.strptime(self.end, "%Y-%m-%d")]
                params = ExpSmoothing_trend_optimization(trial)
                exp = ExponentialSmoothing(train, trend = params["trend"], damped_trend=params["damped_trend"])
                exp_fit = exp.fit(**ExpSmoothing_Fit_Optimization(trial))        
            
            predict = exp_fit.forecast((datetime.strptime(self.end,"%Y-%m-%d") - datetime.strptime(self.start,"%Y-%m-%d")).days+1)
            mae = np.mean(np.abs(test[datetime.strptime(self.start,"%Y-%m-%d"):datetime.strptime(self.end,"%Y-%m-%d")][self.col] - pd.DataFrame(predict,index=test[:datetime.strptime(self.end,"%Y-%m-%d")].index, columns=[self.col]).fillna(0)[self.col]))    

        return mae
    
    def tune_parameters(self):
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial : self.objective(trial), n_trials=self.trial)
        return study


def metrics(train,test):
    metric = []
    metric.append(mean_absolute_error(test, train))
    metric.append(np.sqrt(mean_squared_error(test,train)))
    metric.append(np.sqrt(mean_squared_log_error(test,train)))
    return metric
