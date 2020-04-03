# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:29:23 2020

@author: msi
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