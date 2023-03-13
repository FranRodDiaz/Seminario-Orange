import pandas as pd # Importamos pandas

from Orange.data import Table   # Importamos el dato Table
from orangewidget.widget import OWBaseWidget, Input, Output # Importamos la clase base, la de entrada y salida
from orangewidget.settings import Setting   # Importamos setting para configurar los parametros por defecto
from orangewidget import gui    # Importamos gui para construir el panel de configuración


class OWSortTable(OWBaseWidget):
    
    name = "Sort table"  # Nombre del widget
    description = "Sort the table by one of it's columns"   # Descripcion del widget
    icon = "icons/sort.png" # Icono del widget
    priority = 10   # Prioridad de carga del widget
    
    class Inputs:   # Definimos la clase de entrada con el dato que recibe el widget
        data = Input("Data", Table)

    class Outputs:  # Definimos la clase de salida con el dato que devuelve el widget
        sorted = Output("Sorted table", Table)
        
        
    columns = []    # Definimos el vector que contendrá el nombre de las columnas
    column = Setting(1) # Definimos la columna por la que ordenará, por defecto a 1
    order = Setting(0)  # Definimos si se ordena ascendentemente o descendentemente (por defecto ascendentemente)
    comboColumns = None #Definimos el combobox
    
    def __init__(self):
        super().__init__()    
        self.optionsBox = gui.widgetBox(self.controlArea, "Options")    # Creamos una caja en el widget
        self.radio = gui.radioButtonsInBox(self.optionsBox, self, 'order', btnLabels=["Ascending", "Descending"], callback=self.sortTable)   #Definimos un radioButton para el tipo de ordenacion
        self.comboColumns = gui.comboBox(self.optionsBox, self, "column", label="Column to sort: ", items=self.columns, # Creamos el combobox
                                         callback=self.sortTable)
        self.optionsBox.setDisabled(True)   # Deshabilitamos está ventana por defecto
        
    @Inputs.data         # Decorador para los datos de entrada
    def set_data(self, dataset):     # Metodo para establecer los datos
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
           
    def sortTable(self):    # Metodo para ordenar la tabla por la columna y orden que escoga
        df = pd.DataFrame(self.dataset)     # Transformamos la tabla recibiba en un dataset
      
        ascending = True        # Por defecto ascending es True
        
        if self.order == 1:     # Si hemos seleccionado descendentemente en el radioButton entonces entraremos 
            ascending = False
            
        df = df.sort_values(self.column, ascending=ascending)   # Ordenamos los valores
        
        self.dataset = Table.from_numpy(domain=self.columns, X=df)  # Transformamos el dataframe en una tabla de Orange de nuevo
        
        self.Outputs.sorted.send(self.dataset)  # La enviamos por el canal sorted
          
    