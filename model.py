##########################################################################################################################################
### RFM MODEL ###
##########################################################################################################################################
import plotly.express as px
import statsmodels.api as sm
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('transaction_data.csv', encoding='ISO-8859-1')

##########################################################################################################################################
### Create the Customer Table ###
##########################################################################################################################################

# Calculate Sales Value
data['sales_value'] = data['Quantity'] * data['UnitPrice']

# group columns by customer_id
rfmTable = data.groupby(
    ['CustomerID'], as_index=False
).agg(
    {
        'sales_value' :sum
    ,   'InvoiceNo': pd.Series.nunique
    ,   'InvoiceDate': max
    }
)

# Calculate recency
rfmTable['InvoiceDate'] = pd.to_datetime(rfmTable['InvoiceDate']) 
rfmTable['InvoiceDate'] = rfmTable.InvoiceDate.dt.date

today = rfmTable.InvoiceDate.max() #use the latest date in the dataset - in the real world this will be todays system date
rfmTable['Recency'] = (today - rfmTable['InvoiceDate']).dt.days #Days since last order

##########################################################################################################################################
### Stats Tests ###
##########################################################################################################################################

# first rename the columns to a more user friendly format
rfmTable = rfmTable.rename(columns={
    'sales_value':'MonetaryValue', 'InvoiceNo':'Frequency', 'InvoiceDate':'LastOrderDate'
    }
)

#show distribution of values
#recency
fig = px.histogram(rfmTable, x="Recency", y="CustomerID", marginal="box", # or violin, rug
                   hover_data=rfmTable.columns, title='Recency Plot')
fig.show()

#frequency
fig = px.histogram(rfmTable, x="Frequency", y="CustomerID", marginal="box", # or violin, rug
                   hover_data=rfmTable.columns, title='Frequency Plot')
fig.show()

#monetary value
fig = px.histogram(rfmTable, x="MonetaryValue", y="CustomerID", marginal="box", # or violin, rug
                   hover_data=rfmTable.columns, title='Monetary Value Plot')
fig.show()

#Q-Q plot of the quantiles of x versus the quantiles/ppf of a distribution.
# set up the plot figure
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot as plt
f, axes = plt.subplots(2, 2, figsize=(20,12))

#define distribution graphs
qqplot(rfmTable.Recency, line='r', ax=axes[0,0], label='Recency')
qqplot(rfmTable.Frequency, line='r', ax=axes[0,1], label='Frequency')
qqplot(rfmTable.MonetaryValue, line='r', ax=axes[1,0], label='MonetaryValue')

#plot all
plt.tight_layout()

##########################################################################################################################################
### RFM Score Function ###
##########################################################################################################################################
# Detemine the dataset quantiles
q = np.arange(0, 1, 0.10).tolist()
quantiles = rfmTable.quantile(q=np.around(q,decimals=2))

# Send the quantiles to the dictionary
quantiles = quantiles.to_dict()

# Start creating the RFM segmentation table
rfmSegmentation = rfmTable[['CustomerID','MonetaryValue','Frequency','Recency']]

# We created to classes where high recency is bad and high frequency/ money is good

# 1. Arguments (x = value, work on intervals of 90 days)
def RClass(x):
    if x <= 90:
        return 1
    elif x <= 180:
        return 2
    elif x <= 270: 
        return 3
    elif x <= 360: 
        return 4
    elif x <= 540: 
        return 5    
    else:
        return 6
    
# 2. Arguments (x = value, p = frequency)
def FClass(x,p,d):
    if x <= d[p][0.3]:
        return 6
    elif x <= d[p][0.4]:
        return 5
    elif x <= d[p][0.6]:
        return 4
    elif x <= d[p][0.8]: 
        return 3
    elif x <= d[p][0.9]: 
        return 2
    else:
        return 1
    
# 3. Arguments (x = value, p = monetary_value)
def MClass(x,p,d):
    if x <= d[p][0.2]:
        return 6
    elif x <= d[p][0.4]:
        return 5
    elif x <= d[p][0.6]:
        return 4
    elif x <= d[p][0.8]: 
        return 3
    elif x <= d[p][0.9]: 
        return 2
    else:
        return 1

# 4. Customer Segment Arguments (x = value, slice by value distribution in order to segment stage)

def CustomerSegment(x):
    if x['R_Quartile'] ==1 and x['F_Quartile'] ==1 and x['M_Quartile'] ==1:
        return "Champions"
    elif x['R_Quartile'] <=2 and x['F_Quartile'] <=2 and x['M_Quartile'] <=2:
        return "Loyal_Customers"
    elif x['R_Quartile'] <=2 and x['F_Quartile'] <=3 and x['M_Quartile'] <=3:
        return "Potential_Loyalists"
    elif x['R_Quartile'] <=2 and x['F_Quartile'] <=4 and x['M_Quartile'] <=4:
        return "Promising"
    elif x['R_Quartile'] <=2 and x['F_Quartile'] <=6 and x['M_Quartile'] <=6:
        return "Recent_Customers"
    elif x['R_Quartile'] ==3 and x['F_Quartile'] <=3 and x['M_Quartile'] <=3:
        return "Customer_Needs_Attention"
    elif x['R_Quartile'] ==3 or x['R_Quartile'] ==4 and x['F_Quartile'] >=5 and x['M_Quartile'] >=5:
        return "Hibernating"
    elif x['R_Quartile'] ==4 and x['F_Quartile'] <=3 and x['M_Quartile'] <=3:
        return "At_Risk"
    elif x['R_Quartile'] ==4 and x['F_Quartile'] >=3 and x['M_Quartile'] >=3:
        return "About_to_Sleep"
    elif x['R_Quartile'] >=5 and x['F_Quartile'] >=3 and x['M_Quartile'] >=3:
        return "Lost"
    elif x['R_Quartile'] ==5 and x['F_Quartile'] <=3 and x['M_Quartile'] <=3:
        return "Cant_Lose_Them"
    elif x['R_Quartile'] ==6 and x['F_Quartile'] <=3 and x['M_Quartile'] <=3:
        return "High_Value_Sleeping"
    else:
        return "Lost"

##########################################################################################################################################
### CALCULATE THE RFM SCORES ###
##########################################################################################################################################

# Scores
rfmSegmentation['R_Quartile'] = rfmSegmentation['Recency'].apply(RClass)
rfmSegmentation['F_Quartile'] = rfmSegmentation['Frequency'].apply(FClass, args=('Frequency',quantiles,))
rfmSegmentation['M_Quartile'] = rfmSegmentation['MonetaryValue'].apply(MClass, args=('MonetaryValue',quantiles,))

# Classify the RFM score for the customer base
rfmSegmentation['RFMClass'] = rfmSegmentation.R_Quartile.map(str) \
                            + rfmSegmentation.F_Quartile.map(str) \
                            + rfmSegmentation.M_Quartile.map(str)

# Classify customer segments based on RFM scores

rfmSegmentation['Customer Segment'] = rfmSegmentation.apply(lambda x: CustomerSegment(x), axis=1)

#scatter plot to display segments
rfm_scatter = rfmSegmentation[(rfmSegmentation['MonetaryValue'] > 0) & (rfmSegmentation['Recency'] <=360) & (rfmSegmentation['Frequency'] <= 50)]
fig = px.scatter(rfm_scatter, x="Recency", y="Frequency", color="Customer Segment",
                 size='MonetaryValue', hover_data=['R_Quartile', 'F_Quartile', 'M_Quartile'])
fig.show()

# Save the results to a csv file
output_table = rfmSegmentation.to_csv('rfm_segments.csv')

print('RFM Calculation Completed!')