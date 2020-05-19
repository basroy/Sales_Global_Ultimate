"""
   Created on 18-May-2020
   This is to apply regression techniques to predict sales values
   Metrics available - Contract Dates, Contract duration, Sales territory
@author: Bashobi
"""

import pandas as pd
import numpy as np
import datetime
import os#, sqlalchemy#, sqlparse
from calendar import monthrange
from dateutil.relativedelta import relativedelta

try:
    print(os.path.basename(__file__)+' start '+ str(datetime.datetime.now()))
except:
    print('start '+ str(datetime.datetime.now()))

remote = os.getcwd()=='/home/basroy/scripts/python'
local_path = 'C:/Users/basroy/Desktop/'
remote_path = '/home/basroy/data'
path = remote_path if remote else local_path

#Load data...the sql extraction can be included in this script
deal_data=pd.read_csv(path+'SC3/OTM/Weborder_trend.csv')
deal_data['CONTRACT_START_DATE'] = deal_data['CONTRACT_START_DATE'].fillna('')
deal_data['CONTRACT_END_DATE'] = deal_data['CONTRACT_END_DATE'].fillna('')

deal_data['SALES_CHANNEL_CODE'] = deal_data['SALES_CHANNEL_CODE'].fillna('UNK')#print(deal_data)

measures = ['TRX_USD_VALUE', 
            'TERM',
            'REMAINING_TERM']

for col in measures:
    deal_data[col] = deal_data[col].astype(str).str.replace(',','').astype(float)       

# Need to Calculate Contract End Date, if it is NULL, based on Terms
deal_data['CONTRACT_END_DATE2'] =  pd.to_datetime(deal_data['CONTRACT_END_DATE'])
deal_data['CONTRACT_START_DATE'] = np.where(deal_data['CONTRACT_START_DATE'].isnull(), deal_data['OTM_DATE'], deal_data['CONTRACT_START_DATE'])

deal_data['CONTRACT_START_DATE2'] =  pd.to_datetime(deal_data['CONTRACT_START_DATE'])
deal_data['CONTRACT_START_DATE2'] = deal_data['CONTRACT_START_DATE2'].astype('datetime64[ns]')
#
# measure TERM is in units of months. hence create another column for TERM_in_Days
deal_data['TERM_IN_DAYS'] = np.where(deal_data['TERM'].isnull(), 0 ,
          deal_data.apply(lambda x: x['TERM'] * 30, axis=1) )
# Converting float value to Days , so that it can be added to a datetime()
deal_data['TERM_IN_DAYS_ns'] = pd.to_timedelta(deal_data['TERM_IN_DAYS'], unit='D') 
deal_data['CONTRACT_END_DATE2'] = np.where(deal_data['CONTRACT_END_DATE2'].isna(), 
        deal_data['CONTRACT_START_DATE2'] + deal_data['TERM_IN_DAYS_ns'] , deal_data['CONTRACT_END_DATE2'])         
#print(deal_data['CONTRACT_END_DATE2'].head(10))

