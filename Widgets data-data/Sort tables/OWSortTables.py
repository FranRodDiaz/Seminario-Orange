import numpy as np
import pandas as pd

from Orange.data import Table
from AnyQt.QtCore import Qt
from orangewidget.widget import OWBaseWidget, Input, Output
from orangewidget.utils.widgetpreview import WidgetPreview
from orangewidget.settings import Setting
from orangewidget import gui


class OWSortTables(OWBaseWidget):
    
    name = "Sort tables"
    description = "Sort the table by one of it's columns"
    icon = "sort.png"
    priority = 10
    
    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        sorted = Output("Sorted table", Table)
        
        
    columns = []
    column = Setting(0)
    order = Setting(0)
    comboColumns = None
    
    def __init__(self):
        super().__init__()    
        self.optionsBox = gui.widgetBox(self.controlArea, "Options")
        radio = gui.radioButtonsInBox(self.optionsBox, self, 'order', btnLabels=["Ascending", "Descending"], callback=self.sortTable)
        self.comboColumns = gui.comboBox(self.optionsBox, self, "column", label="Columns to sort: ", items=self.columns, orientation=Qt.Horizontal, 
                                         callback=self.sortTable)
        self.optionsBox.setDisabled(True)
        
    @Inputs.data
    def set_data(self, dataset):
        if dataset is not None:
            self.dataset = dataset
            self.columns = dataset.domain
            i = 0;
            self.comboColumns.clear()
            
            for i in range(len(self.columns)):
                self.comboColumns.addItem(str(self.columns[i]))
        
            self.optionsBox.setDisabled(False)   
            
            self.sortTable()
        else:
            self.dataset = None
            self.Outputs.sorted.send(self.dataset)
           
    def sortTable(self):
        df = pd.DataFrame(self.dataset)
      
        ascending = True
        if self.order == 1:
            ascending = False
        df = df.sort_values(self.column, ascending=ascending)
        self.dataset = Table.from_numpy(domain=self.columns, X=df)
        self.Outputs.sorted.send(self.dataset)
          
    