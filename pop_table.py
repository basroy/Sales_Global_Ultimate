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
        wx.Frame.__init__(self,parent,id,'Bashobi frame',pos=(55,30),size=(500,300))
        panel = wx.Panel(self)
        button = wx.Button(panel, label="exit", pos=(230,110),size=(50,30))
        self.Bind(wx.EVT_BUTTON, self.closebutton,button)
        menuBar = wx.MenuBar()
        filemenu = wx.Menu()
        editmenu = wx.Menu()
        filemenu.Append(wx.ID_FILE1, "New Window","Open new terminal")
        # filemenu.Append(wx.NewId(), "Save","Save file")
        #filemenu.Append(wx.ID_EXIT, "Exit","Next query run")
        exitItem = filemenu.Append(wx.ID_EXIT, "Exit","Next query run")
        menuBar.Append(filemenu, "File")
        menuBar.Append(editmenu, "Edit")
        self.SetMenuBar(menuBar)
        self.Bind(wx.EVT_MENU, self.closebutton, exitItem)
        grid = gridlib.Grid(panel)
        grid.CreateGrid(dfshape[0],dfshape[1])
        box = wx.TextEntryDialog(None, "Enter a SQL Script")
        if box.ShowModal() == wx.ID_OK:
            script = box.GetValue()

    def closebutton(self,event):
        self.Close(True)

    def closewindow(self,event):
        self.Destroy(True)        

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

