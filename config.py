import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso
from statsmodels.nonparametric.smoothers_lowess import lowess

import optuna
import pymannkendall as mk
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from prophet import Prophet


sales = pd.read_csv(r"store-sales-time-series-forecasting\train.csv")
test_sales = pd.read_csv(r"store-sales-time-series-forecasting\test.csv")
oil = pd.read_csv(r"store-sales-time-series-forecasting\oil.csv")
holidays = pd.read_csv(r"store-sales-time-series-forecasting\holidays_events.csv")
stores = pd.read_csv(r"store-sales-time-series-forecasting\stores.csv")

sales.drop(["id"],axis=1,inplace=True)

sales["date"] = sales["date"].apply(lambda x: datetime.strptime(x,"%Y-%m-%d"))

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


def seasonal_iqr(data, freq):
    seasonal_iqrs = []

    for i in range(1, freq + 1):
        seasonal_data = data[data["Day_of_Year"] == i]['sales']
        q1 = seasonal_data.quantile(0.25)
        q3 = seasonal_data.quantile(0.75)
        iqr = q3 - q1
        seasonal_iqrs.append((i, q1 - 1.5 * iqr, q3 + 1.5 * iqr))

    return seasonal_iqrs

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
    def __init__(self, model, params, init_data, start_date, end_date, scaling = False, drop=[]):
        self.model = model
        self.params = params
        self.scaling = scaling
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
        print("Date while creating dataset: ",X.index.max())
        look_back = 365
        X_1 = self.init_data[datetime.strptime(self.start_date, "%Y-%m-%d"):]
        X = self.lagged(X)
        X = self.rolling(X)
        X["sales_expanding_wt"] = X["sales"].expanding().agg(expanding_wt_mean,0.2).shift(1).fillna(method='bfill')
        X["sales_wt_mean7"] = X["sales"].rolling(7).agg(rolling_wt_mean,0.6).shift(1).fillna(method='bfill')
        X = X.drop(self.drop, axis=1)
        return X, X_1

    def construct_train(self,predicted, index, X1, X_1, alp, N=None):
        X1.loc[index,"sales"] = predicted
        next = X1.index.max() + timedelta(days=1)
        print("Date while re-constructing the train: ",next)

        X1.loc[next,"onpromotion"] = X_1.loc[next,"onpromotion"]

        X1.loc[next,"Holiday"] = X_1.loc[next,"Holiday"]
        X1.loc[next,"Additional"] = X_1.loc[next,"Additional"]
        #X1.loc[next,"Lift"] = X_1.loc[next,"Lift"]
        X1.loc[next,"Christmas"] = X_1.loc[next,"Christmas"]
        X1.loc[next,"dcoilwtico_in"] = X_1.loc[next,"dcoilwtico_in"]
        if any("Day_of_Week" in x for x in list(X1.columns)):
            X1.loc[next,"Day_of_Week"] = X_1.loc[next,"Day_of_Week"]
        if any("Week_of_Year" in x for x in list(X1.columns)):
            X1.loc[next,"Week_of_Year"] = X_1.loc[next,"Week_of_Year"]
        if any("Month_of_Year" in x for x in list(X1.columns)):
            X1.loc[next,"Month_of_Year"] = X_1.loc[next,"Month_of_Year"]
        X1 = self.lagged(X1)
        X1 = self.rolling(X1)
        if alp == 999:
            alpha = SimpleExpSmoothing(X1["sales"]).fit(optimized=True).params["smoothing_level"]
        else : alpha = alp
        X1["sales_expanding_wt"] = X1["sales"].expanding().agg(expanding_wt_mean, alpha).shift(1).fillna(method='bfill')

        X1["sales_wt_mean7"] = X1["sales"].rolling(7).agg(rolling_wt_mean, (alpha)).shift(1).fillna(method='bfill')
        X1.drop(self.drop,axis=1,inplace=True)

        return X1

    def fit(self,X, y = None, drop = []):
        last_day = X.index.max() - timedelta(days=1)
        print("Date at the time of fitting: ",last_day)
        mx = MinMaxScaler()
        X_train = X.drop(["sales"],axis=1)
        if self.scaling:
            X_train = pd.DataFrame(mx.fit_transform(X_train), columns=X.drop(["sales"],axis=1).columns, index=X.index)
        else : X_train = pd.DataFrame(X_train, columns=X.drop(["sales"],axis=1).columns, index=X.index)
        ls = self.model()
        ls.set_params(**self.params)
        ls_fit = ls.fit(X_train.loc[:last_day], X.loc[:last_day,"sales"])
        return ls_fit

    def predict(self,X, N=None):
        X, X_1 = self.creating_dataset(N)
        pred1 = []
        mx = MinMaxScaler()
        for i in range(len(X_1[:datetime.strptime(self.end_date,"%Y-%m-%d")])):
            next_day = X_1.index.min() + timedelta(days=i)
            print("Date at the time of prediction: ",next_day)
            ls_fit = self.fit(X[:X_1.index.min() + timedelta(days=i-1)])
            if self.scaling:
                X_train = pd.DataFrame(mx.fit_transform(X.drop(["sales"],axis=1)), columns=X.drop(["sales"],axis=1).columns, index=X.index)
            else : X_train = pd.DataFrame(X.drop(["sales"],axis=1), columns=X.drop(["sales"],axis=1).columns, index=X.index)
            pred = ls_fit.predict(X_train[next_day:next_day])
            pred1.append(pred)
            X = self.construct_train(pred,next_day, X, X_1, alp=0.2, N = N)
        return np.array(pred1)

class Tuning():
    def __init__(self, model, init_data, start, end, col, trial, drop=[],scaling=False):
        self.model = model
        self.scaling = scaling
        self.init_data = init_data
        self.start = start
        self.end = end
        self.drop = drop
        self.trial = trial
        self.col = col


    def rf_optimization(self, trail):
        params = {"max_depth": trail.suggest_int("max_depth",7,59,step=2), "n_estimators":trail.suggest_int("n_estimators",250,1000,step=250),"max_features":trail.suggest_categorical("max_features",["log2","sqrt"]),"min_samples_split":trail.suggest_int("min_samples_split",2,10,step=2),"random_state":42, "n_jobs":-1}
        return params

    def lgb_optimization(self, trail):
        params = {"max_depth": trail.suggest_int("max_depth",7,59,step=2), "learning_rate":trail.suggest_float("learning_rate",0.1,0.8,step=0.1),"bagging_fraction":trail.suggest_float("bagging_fraction",0.8,1,step=0.1),"feature_fraction":trail.suggest_float("feature_fraction",0.8,1,step=0.1), "reg_alpha":trail.suggest_categorical("reg_alpha",[0.1, 0.5, 1, 10]),"verbosity":-1, "random_state":42,"n_jobs":-1}
        return params

    def xgb_optimization(self, trail):
        params = {"max_depth": trail.suggest_int("max_depth",7,69,step=2), "eta":trail.suggest_float("eta",0.1,0.8,step=0.1),"lambda":trail.suggest_categorical("lambda",[0.1, 0.5, 1, 10]),"booster":trail.suggest_categorical("booster",["gbtree"]),"n_jobs":-1, "n_estimators":trail.suggest_categorical("n_estimators",[100, 200, 250, 500]), "subsample":trail.suggest_float("subsample",0.7,1, step=0.1)}
        return params

    def cat_optimization(self, trail):
        params={"iterations":trail.suggest_int("iterations",250,1000,step=250), "grow_policy":trail.suggest_categorical("grow_policy",["Lossguide","Depthwise","SymmetricTree"]), "learning_rate":trail.suggest_float("learning_rate",0.1,0.8,step=0.1), "depth":trail.suggest_int("depth",7,59,step=2), "l2_leaf_reg":trail.suggest_categorical("l2_leaf_reg",[3,4,5,6]), "verbose":0, "random_state":42, "thread_count":-1, "cat_features":["description"]}
        return params

    def objective(self, trial: optuna.Trial):
        if self.model == RandomForestRegressor:
            fcst = Forecasting(RandomForestRegressor,self.rf_optimization(trial),self.init_data,self.start,self.end,drop=self.drop)
        elif self.model == XGBRegressor:
            fcst = Forecasting(XGBRegressor,self.xgb_optimization(trial),self.init_data,self.start,self.end,drop=self.drop)
        elif self.model == CatBoostRegressor:
            fcst = Forecasting(CatBoostRegressor,self.cat_optimization(trial),self.init_data,self.start,self.end,drop=self.drop)
        elif self.model == LGBMRegressor:
            fcst = Forecasting(LGBMRegressor,self.lgb_optimization(trial),self.init_data,self.start,self.end,drop=self.drop)
        X, X1 = fcst.creating_dataset()
        ls_fit = fcst.fit(X)
        predict = fcst.predict(X)

        return np.mean(np.abs(X1[datetime.strptime(self.start,"%Y-%m-%d"):datetime.strptime(self.end,"%Y-%m-%d")][self.col] - pd.DataFrame(predict,index=X1[:datetime.strptime(self.end,"%Y-%m-%d")].index, columns=[self.col]).fillna(0)[self.col]))

    def tune_parameters(self):
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial : self.objective(trial), n_trials=self.trial)
        return study


def metrics(train,test):
    metric = []
    metric.append(mean_absolute_error(test, train))
    metric.append(np.sqrt(mean_squared_error(test,train)))
    return metric