
##########################################################################################################################################
### RFM MODEL ###
##########################################################################################################################################

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
### RFM Score Function ###
##########################################################################################################################################

# first rename the columns to a more user friendly format
rfmTable = rfmTable.rename(columns={
    'sales_value':'MonetaryValue', 'InvoiceNo':'Frequency', 'InvoiceDate':'LastOrderDate'
    }
)

# Detemine the dataset quantiles
q = np.arange(0, 1, 0.10).tolist()
quantiles = rfmTable.quantile(q=np.around(q,decimals=2))

# Send the quantiles to the dictionary
quantiles = quantiles.to_dict()

# Start creating the RFM segmentation table
rfmSegmentation = rfmTable[['CustomerID','MonetaryValue','Frequency','Recency']]

# We created to classes where high recency is bad and high frequency/ money is good

# 1. Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
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
    
# 2. Arguments (x = value, p = recency, frequency)
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
    
# 3. Arguments (x = value, p = recency, monetary_value, frequency)
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

# 4. Customer Segment Arguments (x = value, a = recency, b = frequency, c = monetary_value)

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

# Save the results to a csv file
output_table = rfmSegmentation.to_csv('rfm_segments.csv')

print('RFM Calculation Completed!')