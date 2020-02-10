import pandas as pd
import cx_Oracle
import wx
import wx.grid as gridlib

conn = cx_Oracle.connect('XXG2CRO/C1sc0bea7@G2CPRD')
cursor = conn.cursor()
cursor.execute('Select trx_id, trx_number, Source_trx_id, otm_date, creation_date, last_update_date, deal_id \
                   , trx_usd_value  , source_trx_version_id , otm_batch_id, trx_applied_line_id \
                   from xxotm.xxotm_phx_trx where rownum <= 10')
records = cursor.fetchall()
df = pd.DataFrame(records)
dfshape = df.shape
griddata = df.transpose()
#print(df)
print(dfshape[0],dfshape[1])
conn.close()

EVEN_ROW_COLOUR = '#CCE6DD'
GRID_LINE_COLOUR = '#aaa'

app = wx.App()
frame = wx.Frame(None,title='Bashobi frame',pos=(25,50),size=(500,300))
panel = wx.Panel(frame)
button = wx.Button(panel, label="exit", pos=(130,10),size=(20,20)
                   # Trying to not use Class....but dont see a way. 

menuBar = wx.MenuBar()
fileButton = wx.Menu()
editButton = wx.Menu()
menuBar.Append(fileButton, 'File')
menuBar.Append(editButton, 'Edit')

grid = gridlib.Grid(panel)
grid.CreateGrid(dfshape[0],dfshape[1]) #change to make all data from datagrid fit, makew this fit any query
sizer = wx.BoxSizer(wx.VERTICAL)
sizer.Add(grid, 1,wx.EXPAND)
panel.SetSizer(sizer)

#Populate the Panel from df
#table = wx.grid.PyGridTableBase(df)

griddata.aggregate
frame.Show()
app.MainLoop()
print(len(griddata))
print(len(griddata.columns))
