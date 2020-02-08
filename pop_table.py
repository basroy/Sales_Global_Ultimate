import pandas as pd
import cx_Oracle
import wx
import wx.grid as gridlib

conn = cx_Oracle.connect('XXG2CRO/C1sc0bea7@G2CPRD')
cursor = conn.cursor()
cursor.execute('Select * from xxotm.xxotm_phx_trx where rownum <= 10')
records = cursor.fetchall()
df = pd.DataFrame(records)
griddata = df.transpose()
print(df)
conn.close()
app = wx.App()
frame = wx.Frame(None,title='Bashobi frame',pos=(25,150),size=(300,400))
panel = wx.Panel(frame)
grid = gridlib.Grid(panel)
grid.CreateGrid(7,5) #change to make all data from datagrid fit, makew this fit any query
sizer = wx.BoxSizer(wx.VERTICAL)
sizer.Add(grid, 1,wx.EXPAND)
panel.SetSizer(sizer)
frame.Show()
app.MainLoop()
print(len(griddata))
print(len(griddata.columns))