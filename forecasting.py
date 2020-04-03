# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:01:02 2020

@author: abc
"""
# COCACOLA SALES
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

cocacola=pd.read_excel("F:\\Data Science\\Assignemnts\\Brindan\\Forecasting\\CocaCola_Sales_Rawdata.xlsx")
np.mean(cocacola.Sales)
cocacola.columns
print(cocacola.Quarter)
cocacola.describe()
pd.crosstab(cocacola.Quarter, cocacola.Sales)
sb.boxplot(cocacola.Sales, orient='v')
plt.hist(cocacola.Sales)

quarter =['Q1','Q2','Q3','Q4'] 
p = cocacola["Quarter"][0]
p[0:2]
cocacola['quarters']= 0

for i in range(42):
    p = cocacola["Quarter"][i]
    cocacola['quarters'][i]= p[0:2]
    
quarter_dummies = pd.DataFrame(pd.get_dummies(cocacola['quarters']))
cocacola1 = pd.concat([cocacola,quarter_dummies],axis = 1)

cocacola1["t"] = np.arange(1,43)

cocacola1["t_squared"] = cocacola1["t"]*cocacola1["t"]
cocacola1.columns
cocacola1["log_Quarter"] = np.log(cocacola1["Sales"])

cocacola1.Sales.plot()
Train = cocacola1.head(38)
Test = cocacola1.tail(4)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13))

#L I N E A R Model
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear

# Exponential

Exp = smf.ols('log_Quarter~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

# Quadratic 

Quad = smf.ols('Sales~t+t_square',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad

# Additive seasonality 

add_sea = smf.ols('Sales~Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1','Q2','Q3','Q4']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea

# Additive Seasonality Quadratic 

add_sea_Quad = smf.ols('Sales~Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1','Q2','Q3','Q4','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

# Multiplicative Seasonality 

Mul_sea = smf.ols('log_Quarter~Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

#Multiplicative Additive Seasonality 

Mul_Add_sea = smf.ols('log_Quarter~t+Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

#Testing 

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse


# i will go with linear additive seasonality 





# Plastic Sales
plastic=pd.read_csv("F:\\Data Science\\Assignemnts\\Brindan\\Forecasting\\PlasticSales.csv")
plastic.columns
month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
#import numpy as np
p = plastic["Month"][0]
p[0:3]
plastic['months']= 0

for i in range(60):
    p = plastic["Month"][i]
    plastic['months'][i]= p[0:3]
    
month_dummies = pd.DataFrame(pd.get_dummies(plastic['months']))
plastic1 = pd.concat([plastic,month_dummies],axis = 1)

plastic1["t"] = np.arange(1,61)

plastic1["t_squared"] = plastic1["t"]*plastic1["t"]
plastic1.columns
plastic1["log_sales"] = np.log(plastic1["Sales"])
plastic1.Sales.plot()
Train = plastic1.head(47)
Test = plastic1.tail(12)

# L I N E A R 
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear

# Exponential 

Exp = smf.ols('log_sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

# Quadratic

Quad = smf.ols('Sales~t+t_square',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad

# Additive seasonality 

add_sea = smf.ols('Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea

# Additive Seasonality Quadratic 

add_sea_Quad = smf.ols('Sales~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

# Multiplicative Seasonality 

Mul_sea = smf.ols('log_salesr~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Ridership'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

#Multiplicative Additive Seasonality 

Mul_Add_sea = smf.ols('log_sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Ridership'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

# Testing 

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

# I will go with linear model 







# Airlines
airlines=pd.read_excel("F:\\Data Science\\Assignemnts\\Brindan\\Forecasting\\Airlines+Data.xlsx")
airlines.columns
print(airlines.Month)
airlines['Month'] = airlines.Month.astype('str')
months =['01','02','03','04','05','06','07','08','09','10','11','12'] 
p = airlines["Month"][0]
p[5:7]
airlines['months']= 0

for i in range(96):
    p = airlines["Month"][i]
    airlines['months'][i]= p[5:7]
    
month_dummies = pd.DataFrame(pd.get_dummies(airlines['months']))
airlines1 = pd.concat([airlines,month_dummies],axis = 1)
airlines1.columns
airlines1["t"] = np.arange(1,97)

airlines1["t_squared"] = airlines1["t"]*airlines1["t"]
airlines1.columns
airlines1["log_passengers"] = np.log(airlines["Passengers"])
airlines1.columns
airlines1.Passengers.plot()
Train = airlines1.head(84)
Test = airlines1.tail(12)

#Multiplicative Additive Seasonality 

Mul_Add_sea = smf.ols('log_passenger~t+1+2+3+4+5+6+7+8+9+10+11+12',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 


# I will go with Multiplicative Additive Seasonality 