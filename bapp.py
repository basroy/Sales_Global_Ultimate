from flask import Flask, render_template, request
import sqlite3
import json
import os
import pandas as pd
import csv
import datetime
import time
from ast import literal_eval
from pandas_highcharts.core import serialize

bapp = Flask(__name__)
 
"""
   DB created and populated:::
   import sqlite3
sqlite_file = 'db.sqlite'
conn = sqlite3.connect(sqlite_file)
c = conn.cursor()
conn.execute("CREATE TABLE measures (timestamp DATETIME, measure INTEGER)")
conn.commit()
conn.close()

import sqlite3
import time
from random import randint
 
sqlite_file = 'db.sqlite'
timestamp_begin = 1388534400  # 06/12/20 11:00
#timestamp_end = timestamp_begin +  60*100
timestamp_end = 1451520000
pitch = 3600
 
try:
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()
    timestamp = timestamp_begin
    while timestamp <= timestamp_end:
        print("Iterations left :", (timestamp_end-timestamp)/pitch)
        measure = randint(0, 9)
        conn.execute("INSERT INTO measures (timestamp, measure) VALUES ({timestamp}, {measure})".format(timestamp=timestamp, measure=measure))
        conn.commit()
        timestamp += pitch
except Exception as e:
    conn.rollback()
    raise e
finally:
    conn.close()
"""
basedir = os.path.abspath(os.path.dirname(__file__))
print('path returned is ' + basedir) 
remote = os.getcwd()=='/home/basroy/scripts/python'
local_path = 'C:/Users/basroy/Desktop/'
remote_path = '/home/basroy/data'
path = remote_path if remote else local_path

def num(s):
        try:
            return int(s)
        except ValueError:
            return float(s)

@bapp.route("/data.json")
def data():
    connection = sqlite3.connect("db.sqlite")
    cursor = connection.cursor()
    cursor.execute("SELECT 1000*timestamp, measure from measures")
    results = cursor.fetchall()
    #print(results) 
    
    res = []
    inputfile = path+'SC3/OTM/Weborder_highchart.csv'
    with open(inputfile, "r", encoding='utf-8-sig') as ifile:
        res=[tuple(csvrow) for csvrow in csv.reader(ifile, delimiter=",", skipinitialspace=True)]
        #print(res)
        df = pd.DataFrame(res)
        #print(df)
        df.columns = ['SalesValue', 'Discount', 'Offer_Type', 'Initial_Term', 'Remaining_Term', 'Contract_Start_Date', 'Contract_End_Date']
        df['SalesValue'] = df['SalesValue'].astype(float) 
        df = df[df['SalesValue'] > 170000]
        #df = df['Discount'].fillna(0)
        df['Discount'] = df['Discount'].astype(float)
        df1 = pd.DataFrame.from_records(df.values)  
        res = list(df1.itertuples(index=False, name=None)) 
        
    return json.dumps(res)
 
@bapp.route("/graph")
def graph():
    return render_template('graph.html')
 
 
if __name__ == '__main__':
    bapp.run(
    debug=True,
    threaded=True,
    host='127.0.0.1'
)