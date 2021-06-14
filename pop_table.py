import pandas as pd
import cx_Oracle
import wx
import wx.grid as gridlib


class dsptrx(wx.Frame):

    def __init__(self,parent,id):
        wx.Frame.__init__(self,parent,id,'Bashobi frame2',pos=(55,30),size=(500,300))
        panel = wx.Panel(self)
        panel.SetBackgroundColour("gray")
        button = wx.Button(panel, label="exit", pos=(230,110),size=(50,30))
        box = wx.TextEntryDialog(None, "Enter a SQL Script")
        if box.ShowModal() == wx.ID_OK:
            script = box.GetValue()

        buttonEXIT = wx.Button(panel, label="exit", pos=(230,110),size=(50,30))
        self.Bind(wx.EVT_BUTTON, self.closebutton,buttonEXIT)

        menuBar = wx.MenuBar()
        filemenu = wx.Menu()
        editmenu = wx.Menu()
        filemenu.Append(wx.ID_FILE1, "New Window","Open new terminal")
        exitItem = filemenu.Append(wx.ID_EXIT, "Exit","Next query run")
        menuBar.Append(filemenu, "File")
        menuBar.Append(editmenu, "Edit")
        
        sizerMENU =  wx.BoxSizer(wx.HORIZONTAL)
        sizerMENU.Add(menuBar, 0, wx.CENTER|wx.ALL, 5)
 
        #self.SetMenuBar(menuBar)
       # self.Bind(wx.EVT_MENU, self.closebutton, exitItem)
        
        df = self.trxquery()
        #print(df)
        dfshape = df.shape
        griddata = df.transpose()
        grid = gridlib.Grid(panel)
        grid.CreateGrid(dfshape[0],dfshape[1])
        EVEN_ROW_COLOUR = '#CCE6DD'
        GRID_LINE_COLOUR = '#aaa'
        grid.SetColLabelValue(0, "TRX_ID")
        grid.SetColLabelValue(1, "TRX_NUMBER")
        grid.SetColLabelValue(2, "SOURCE_TRX_ID")
        grid.SetColLabelValue(3, "OTM_DATE")
        grid.SetColLabelValue(4, "CREATION_DATE")
        grid.SetColLabelValue(5, "LAST_UPDATE_DATE")
        grid.SetColLabelValue(6, "DEAL_ID")
        grid.SetColLabelValue(7, "TRX_USD_VALUE")
        grid.SetColLabelValue(8, "SOURCE_TRX_VERSION_ID")
        grid.SetColLabelValue(9, "OTM_BATCH_ID")
        grid.SetColLabelValue(10, "TRX_APPLIED_LINE_ID")

        buttonqry  = wx.Button(panel, label="Query Results export")
        sizerqry =  wx.BoxSizer(wx.HORIZONTAL)
        sizerqry.Add(buttonqry, 0, wx.CENTER|wx.ALL, 5)
        sizerqry.Add(buttonEXIT, 0, wx.CENTER|wx.ALL, 5)
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
        conn = cx_Oracle.connect('G2C/something@G2CPRD')
        cursor = conn.cursor()
        cursor.execute('Select trx_id, trx_number, Source_trx_id, otm_date, creation_date, last_update_date, deal_id \
                   , trx_usd_value  , source_trx_version_id , otm_batch_id, trx_applied_line_id \
                   from xxotm.xxotm_phx_trx where rownum <= 10')
        records = cursor.fetchall()
        df = pd.DataFrame(records)
        conn.close()
        return df
       

if __name__ == '__main__':
    app = wx.App()
    frame = dsptrx(parent=None, id=-1)
    frame.Show()
    app.MainLoop()
