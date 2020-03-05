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
#print(griddata)
#print(dfshape[0],dfshape[1])
conn.close()

EVEN_ROW_COLOUR = '#CCE6DD'
GRID_LINE_COLOUR = '#aaa'

class DataPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent = parent)

class dsptrx(wx.Frame):

    def __init__(self,parent,id):
        wx.Frame.__init__(self,parent,id,'Bashobi frame2',pos=(55,30),size=(500,300))
        panel = wx.Panel(self)
        panel.SetBackgroundColour("gray")
        button = wx.Button(panel, label="exit", pos=(230,110),size=(50,30))
        box = wx.TextEntryDialog(None, "Enter a SQL Script")
        if box.ShowModal() == wx.ID_OK:
            script = box.GetValue()
        df = self.trxquery()
        print(df)
        dfshape = df.shape
        griddata = df.transpose()
        grid = gridlib.Grid(panel)
        grid.CreateGrid(dfshape[0],dfshape[1])

        buttonqry  = wx.Button(panel, label="Query Results")
        sizerqry =  wx.BoxSizer(wx.HORIZONTAL)
        sizerqry.Add(buttonqry, 0, wx.CENTER|wx.ALL, 5)
        sizerqry.AddSpacer(10)
        
        sizerdata = wx.BoxSizer(wx.VERTICAL)
        sizerdata.Add(grid, 1,wx.EXPAND)
        sizerdata.Add(sizerqry)
        sizerdata.AddSpacer(10)
        panel.SetSizer(sizerdata)
        columns = dfshape[0]
        rows = dfshape[1]
        for i in range(0,columns):
           for j in range(0,rows):
                grid.SetCellValue(i,j,str(df[j][i]))

    def closebutton(self,event):
        self.Close(True)
        self.Destroy() 

    def closewindow(self,event):
        self.Destroy(True)        

    def trxquery(self):
        conn = cx_Oracle.connect('XXG2CRO/C1sc0bea7@G2CPRD')
        cursor = conn.cursor()
        cursor.execute('Select trx_id, trx_number, Source_trx_id, otm_date, creation_date, last_update_date, deal_id \
                   , trx_usd_value  , source_trx_version_id , otm_batch_id, trx_applied_line_id \
                   from xxotm.xxotm_phx_trx where rownum <= 10')
        records = cursor.fetchall()
        df = pd.DataFrame(records)
        return df
       

if __name__ == '__main__':
    app = wx.App()
    frame = dsptrx(parent=None, id=-1)
    frame.Show()
    app.MainLoop()
    

app = wx.App()
frame = wx.Frame(None,title='Bashobi frame',pos=(25,50),size=(500,300))
panel = wx.Panel(frame)

grid = gridlib.Grid(panel)
grid.CreateGrid(dfshape[0],dfshape[1]) #change to make all data from datagrid fit, makew this fit any query

columns = dfshape[0]
rows = dfshape[1]
for i in range(0,columns):
    for j in range(0,rows):
        grid.SetCellValue(i,j,str(df[j][i]))

sizer = wx.BoxSizer(wx.VERTICAL)
sizer.Add(grid, 1,wx.EXPAND)
panel.SetSizer(sizer)

griddata.aggregate
frame.Show()
app.MainLoop()

