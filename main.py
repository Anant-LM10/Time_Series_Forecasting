import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from config import *

sales = pd.read_csv(r"store-sales-time-series-forecasting\train.csv")
test_sales = pd.read_csv(r"store-sales-time-series-forecasting\test.csv")
oil = pd.read_csv(r"store-sales-time-series-forecasting\oil.csv")
holidays = pd.read_csv(r"store-sales-time-series-forecasting\holidays_events.csv")
stores = pd.read_csv(r"store-sales-time-series-forecasting\stores.csv")

sales.drop(["id"],axis=1,inplace=True)

sales_seg = sales.dropna(axis=0)
sales_seg["tsid"] = sales_seg["family"] + "_" + sales_seg["store_nbr"].astype(int).astype(str)

segment = []

for i in list(sales_seg["tsid"].unique()):
    sales1 = sales_seg[sales_seg["tsid"] == i]
    seg = Segmentation(sales1)
    segment.append([i,seg.segmentation(),seg.trend()])

classif = pd.DataFrame(segment, columns=["tsid","Segment","trend"])
classif[["family","store"]] = classif["tsid"].str.split("_",expand=True)
classif["store"] = classif["store"].astype(int)
classif = sales.reset_index().groupby(["family","store_nbr"])["sales"].sum().reset_index().merge(classif, how="left", left_on=["family","store_nbr"], right_on=["family","store"])
classif["total_sales"] = classif["sales"]
classif = classif.merge(classif.set_index(["store"]).groupby(["store"]).agg({"total_sales":"cumsum"}).reset_index(drop=True), left_index=True, right_index=True)
classif = classif.merge(classif.groupby(["store"])["total_sales_x"].sum().reset_index(), left_on=["store"], right_on=["store"])
classif = classif[["tsid","Segment","trend","family","store","total_sales_x_x","total_sales_y","total_sales_x_y"]]
classif.columns = ["tsid","Segment","trend","sales","categ","store","cumsum_sales","total_sales"]
classif["%_Sales"] = classif["cumsum_sales"]/classif["total_sales"]
classif["ABC_classif"] = np.where(classif["%_Sales"] <= 0.85, "A", np.where(classif["%_Sales"] > 0.95, "C", "B"))


test_sales["date"] = test_sales["date"].apply(lambda x: datetime.strptime(x,"%Y-%m-%d"))
test_sales.set_index("date", inplace=True)
test_sales["tsid"] = test_sales["family"]+"_"+test_sales["store_nbr"].astype(str)
test_sales.drop(["id"],axis=1,inplace=True)

def forecast_concurrent(tsid,acc_rf,acc_lgb,acc_xgb,start,end): 
    
    print("TSID: ",tsid)
    prep = preprocessing(sales, tsid, list(classif[classif["tsid"] == tsid]["trend"])[0])
    sales_tsid, trend_tsid = prep.detrending()
    
    #prep1 = preprocessing(pd.concat((sales, test_sales)), tsid, list(classif[classif["tsid"] == tsid]["trend"])[0])
    #sales_tsid = prep1.final_data()
    sales_tsid["Week_of_Year"] = sales_tsid["Week_of_Year"].astype(int)
    
    #sales_tsid.loc[:,"sales"] = np.array(pd.concat((sales_tsid[:trend_tsid.index.max()]["sales"] - trend_tsid, pd.DataFrame([0]*(sales_tsid.index.max() - trend_tsid.index.max()).days, index=pd.date_range(trend_tsid.index.max()+timedelta(days=1),sales_tsid.index.max())))))
    
    print("Preprocessing completed for tsid: ",tsid)
    
    for i in [RandomForestRegressor, LGBMRegressor, XGBRegressor]:
        print(f"Tuning Started for: {tsid}, model: {i}")
        
        tune = Tuning(i,sales_tsid,tsid,start,end,"sales",1,False,0)
        obj = tune.tune_parameters()
        sales_tsid.drop(["tsid"],axis=1,inplace=True)
        
        print(f"Forecasting started for tsid: {tsid}, model: {i}")
        
        fcst = Forecasting(i, obj.best_params, sales_tsid, start, end)
        X, X_1 = fcst.creating_dataset()
        l1 = fcst.predict(X)

        if len(trend_tsid) > 0:
            day = (datetime.strptime(end,"%Y-%m-%d") - datetime.strptime(start,"%Y-%m-%d")).days + 1
            exp = ExponentialSmoothing(trend_tsid,trend="add",seasonal_periods=365)
            exp_fit = exp.fit()
            trend = exp_fit.forecast(day)
            final_pred = pd.DataFrame(l1, index = sales_tsid[start:end].index)[0] + pd.DataFrame(list(trend), index=sales_tsid[start:end].index, columns=["sales"])["sales"]  
        else :
            final_pred = sales_tsid[start:end][["sales"]]
        
        final_pred = pd.DataFrame(final_pred,columns = ["Prediction"])
        print(f"Forecasting complete for tsid: {tsid}, model: {i}")
        
        print(f"Metrics Calculation started for tsid: {tsid}, model: {i}")       
        if i == RandomForestRegressor:
            final_rf = pd.concat((final_rf, final_pred))
            tmp2 = sales_seg[sales_seg["tsid"] == tsid][["sales"]][start:end].merge(final_pred["Prediction"], left_index=True, right_index=True)[:-1]
            acc_rf = pd.concat((acc_rf, pd.DataFrame([[tsid,1 - np.abs(tmp2["sales"] - tmp2["Prediction"]).sum()/tmp2["sales"].sum(), metrics(tmp2[["sales"]][start:end],tmp2[["Prediction"]])[0], metrics(tmp2[["sales"]][start:end],tmp2[["Prediction"]])[1], metrics(tmp2[["sales"]][start:end],tmp2[["Prediction"]])[2], "RF", obj.best_params]], columns=["tsid","Accuracy","MAE","RMSE","RMSLE","Model","Params"]))).reset_index(drop=True)
        elif i == LGBMRegressor:
            final_lgb = pd.concat((final_lgb, final_pred))
            tmp2 = sales_seg[sales_seg["tsid"] == tsid][["sales"]][start:end].merge(final_pred["Prediction"], left_index=True, right_index=True)[:-1]
            acc_lgb = pd.concat((acc_lgb, pd.DataFrame([[tsid,1 - np.abs(tmp2["sales"] - tmp2["Prediction"]).sum()/tmp2["sales"].sum(), metrics(tmp2[["sales"]][start:end],tmp2[["Prediction"]])[0], metrics(tmp2[["sales"]][start:end],tmp2[["Prediction"]])[1], metrics(tmp2[["sales"]][start:end],tmp2[["Prediction"]])[2], "LGB", obj.best_params]], columns=["tsid","Accuracy","MAE","RMSE","RMSLE","Model","Params"]))).reset_index(drop=True)
        elif i == XGBRegressor:
            final_xgb = pd.concat((final_xgb, final_pred))
            tmp2 = sales_seg[sales_seg["tsid"] == tsid][["sales"]][start:end].merge(final_pred["Prediction"], left_index=True, right_index=True)[:-1]
            acc_xgb = pd.concat((acc_xgb, pd.DataFrame([[tsid,1 - np.abs(tmp2["sales"] - tmp2["Prediction"]).sum()/tmp2["sales"].sum(), metrics(tmp2[["sales"]][start:end],tmp2[["Prediction"]])[0], metrics(tmp2[["sales"]][start:end],tmp2[["Prediction"]])[1], metrics(tmp2[["sales"]][start:end],tmp2[["Prediction"]])[2], "XGB", obj.best_params]], columns=["tsid","Accuracy","MAE","RMSE","RMSLE","Model","Params"]))).reset_index(drop=True)
        print(f"Metrics Calculation complete for tsid: {tsid}, model: {i}")
        
    return acc_rf, acc_lgb, acc_xgb

acc_rf,acc_lgb,acc_xgb = pd.DataFrame()
fcst = Parallel(n_jobs=-1)(delayed(forecast_concurrent)(i,acc_rf,acc_lgb,acc_xgb,"2017-08-01","2017-08-15") for i in list(classif[(classif["ABC_classif"].isin(["A","B"])) & (classif["Segment"].isin(["Smooth","Erratic"]))]["tsid"]))

met_exp = pd.DataFrame()
for tsid in list(classif[(classif["Segment"].isin(["Smooth","Erratic"]))]["tsid"]):

    pre = preprocessing(sales,tsid,classif[classif["tsid"] == tsid]["trend"])
    sales_tsid = pre.final_data()
    sd = MSTL(sales_tsid["sales"], periods=[7,365])
    sd_fit = sd.fit()
    
    seas_365 = pd.DataFrame(sd_fit.seasonal["seasonal_365"])
    seas_365.columns = ["sales"]
    seas_7 = pd.DataFrame(sd_fit.seasonal["seasonal_7"])
    seas_7.columns = ["sales"]
    seas_trend = pd.DataFrame(sd_fit.trend.dropna())
    seas_trend.columns = ["sales"]
    
    best_params = list()
    tune = Tuning(ExponentialSmoothing,seas_365, tsid,"2017-08-01","2017-08-15","sales",1, False,365)
    obj = tune.tune_parameters()
    best_params.append(obj.best_params) 
    tune = Tuning(ExponentialSmoothing,seas_7, tsid,"2017-08-01","2017-08-15","sales",1, False,7)
    obj = tune.tune_parameters()
    best_params.append(obj.best_params)
    tune = Tuning(ExponentialSmoothing,seas_trend, tsid,"2017-08-01","2017-08-15","sales",1, True,0)
    obj = tune.tune_parameters()
    best_params.append(obj.best_params)
    
    fcst = Stat_Models(ExponentialSmoothing, sales, tsid, "2017-08-01","2017-08-15", best_params, classif[classif["tsid"] == tsid]["trend"], False)
    X = fcst.exponential_smoothing()
    met_exp = pd.concat((met_exp,pd.DataFrame([[tsid,1 - np.abs(sales_tsid["sales"]["2017-08-01":"2017-08-15"] - pd.DataFrame(X, index=sales_tsid["sales"]["2017-08-01":"2017-08-15"].index, columns=["sales"])["sales"]).sum()/sales_tsid["sales"]["2017-08-01":"2017-08-15"].sum(), metrics(sales_tsid["sales"]["2017-08-01":"2017-08-15"],pd.DataFrame(X, index=sales_tsid["sales"]["2017-08-01":"2017-08-15"].index, columns=["Prediction"]))[0], metrics(sales_tsid["sales"]["2017-08-01":"2017-08-15"],pd.DataFrame(X, index=sales_tsid["sales"]["2017-08-01":"2017-08-15"].index, columns=["Prediction"]))[1], metrics(sales_tsid["sales"]["2017-08-01":"2017-08-15"],pd.DataFrame(X, index=sales_tsid["sales"]["2017-08-01":"2017-08-15"].index, columns=["Prediction"]))[2], "TES", obj.best_params]], columns=["tsid","Accuracy","MAE","RMSE","RMSLE","Model","Params"])))

met_pr = pd.DataFrame()
for tsid in list(classif[(classif["Segment"].isin(["Smooth","Erratic"]))]["tsid"]):

    tune = Tuning(Prophet, sales, tsid, "2017-08-01", "2017-08-15", "sales", 1, False, 0)
    obj = tune.tune_parameters()
    fcst = Stat_Models(Prophet, sales, tsid, "2017-08-01", "2017-08-15",obj.best_params, False)
    X1 = fcst.prophet_concurrent()
    met_pr = pd.concat((met_pr,pd.DataFrame([[tsid,1 - np.abs(sales_tsid["sales"]["2017-08-01":"2017-08-15"] - pd.DataFrame(X1, index=sales_tsid["sales"]["2017-08-01":"2017-08-15"].index, columns=["sales"])["sales"]).sum()/sales_tsid["sales"]["2017-08-01":"2017-08-15"].sum(), metrics(sales_tsid["sales"]["2017-08-01":"2017-08-15"],pd.DataFrame(X1, index=sales_tsid["sales"]["2017-08-01":"2017-08-15"].index, columns=["Prediction"]))[0], metrics(sales_tsid["sales"]["2017-08-01":"2017-08-15"],pd.DataFrame(X1, index=sales_tsid["sales"]["2017-08-01":"2017-08-15"].index, columns=["Prediction"]))[1], metrics(sales_tsid["sales"]["2017-08-01":"2017-08-15"],pd.DataFrame(X1, index=sales_tsid["sales"]["2017-08-01":"2017-08-15"].index, columns=["Prediction"]))[2], "TES", obj.best_params]], columns=["tsid","Accuracy","MAE","RMSE","RMSLE","Model","Params"])))

    
