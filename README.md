# Time_Series_Forecasting
# Kaggle Competition - Store Sales Forecasting - RMSLE - 0.575
                                                        
Project Overview: 

The case of this competition is to forecast daily sales for an Ecuadorian Superstore. The objective is to predict sales for the next 16 days.

Dataset Description: 

Source: Kaggle - 
Description: The dataset consists of daily sales data from 2013 to Aug - 2017 for various stores.
Number of Families/SKU Categories: 33
Number of Stores: 54

The Dataset shared in the competition had the following info: 

	Sales data with Date, Store, Family, onpromotions and Sales.
	Holiday and Event information with specific information if the holiday is transferred to another date, along with locale and holiday name.
	Oil information: Since the prices of various items is dependent on the oil (for transportation)
	Store information: To which locale the store belongs to and to which category.

Exploratory Data Analysis
	Key insights from EDA was that each family had different nature of sales in each store:
		Trend: Families had a mix of trend patterns varying with each store/locale. 
		Seasonality: Sales peak in December due to holiday shopping, also a usual weekly seasonality was observed in most cases.

Methodology
	
 	Data Preprocessing:

		Filled missing values using interpolation for Oil data.
		Holiday data was restructured to get the exact dates with Holiday and Events.
		Performed Demand Segmentation to classify Family x Store intersections into Smooth, Erratic, Lumpy and Intermittent based on sales.
		Classified Family x Store intersections trend into : increasing, decreasing or no trend.  
	
 	Feature Engineering:
	
  		Merged Holiday, onpromotions and Oil data with the main Sales data. 
		Created Date features - Day_of_Week, Week_of_Year, Month_of_Year.
		For Tree Based Models: 
		Created lagged and rolling features of sales data – sales_lag1, sales_lag7, sales_lag365, expanding wt mean and wt. rolling mean for lag 7.
		Since, tree based models cannot predict the trend, sales were detrended where the trend was present.   
	
 	Models Used:
		
  		Models were decided based on the Segment (Smooth/Erratic/Lumpy) assigned to the intersection. 
		Tree based models - RF, XGB & LGB were tried on Smooth & Erratic data. 
		Stat Models - TES, Prophet (Combination of both) were used on Erratic & Lumpy data.
		ARIMA was also used in combination on Lumpy data.    

	Hyperparameter Tuning: 
		
  		For Hyperparameter Tuning, I used the optuna library which uses “Bayesian Optimization” of Hyperparameters which is faster than the standard techniques like GridSearchCV. 

Model Evaluation:
	
 	Competition Evaluation Metric: RMSLE

Results and Evaluation
	
 	Best Model: Since this was a tournamenting of Models, among the aforementioned models were selected, most prominently TES (38.4% times selected, 18.2% RF and so on). 
	RMSLE: The overall performance of the best fit models was 0.575.

Technologies Used
	
 	Python
	Libraries: Pandas, Numpy, Matplotlib, optuna, statsmodels, prophet, RandomForestRegressor, LGBMRegressor, XGBRegressor, joblib, datetime and pymannkendall.  
