import numpy as np   # Importamos la librería numpy
import Orange.data  # Importamos los datos de Orange
from orangewidget.widget import OWBaseWidget, Input, Output # Importamos la clase base, la de entrada y salida
from orangewidget.settings import Setting   # Importamos setting para configurar los parametros por defecto
from orangewidget import gui    # Importamos gui para construir el panel de configuración
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

class OWFeatureSelection(OWBaseWidget):
    name = "Feature selection"       # Nombre del widget
    description = "" # Descripcion del widget
    icon = "icons/selectAttributes.png" # Icono del widget
    priority = 10   # Prioridad de carga del widget

    class Inputs:           # Definimos la clase de entrada con el dato que recibe el widget
        data = Input("Data", Orange.data.Table)

    class Outputs:          # Definimos la clase de salida con los datos que devuelve el widget
        reducedData = Output("Reduced Data", Orange.data.Table)
        data = Output("Data", Orange.data.Table)

    c = Setting(0.01)
    penalty = Setting("l1")
    dual = Setting(0)
    penaltyLabel = Setting(0)
    #want_main_area = True

    penaltyList = ["l1", "l2"]
    
    def __init__(self):
        super().__init__()

        # GUI
        self.box = gui.widgetBox(self.controlArea, "Options")       # Creamos una caja en el widget

        gui.doubleSpin(self.box, self, 'c', minv=-1, maxv=1,                   # Agregamos un spinBox para elegir el numero del parametro de regularización
                 step=0.01, label='C:', callback = self.selection)
        
        self.kernel_combo = gui.comboBox(   # añadimos un comboBox para seleccionar el tipo de penalty que se utilizará
            self.box, self, "penaltyLabel", label="Penalty type: ", items=self.penaltyList, callback = [self.set_penalty, self.selection]
        )
        
        self.copy = gui.checkBox(    # Agregamos el checkBox para seleccionar el dual (NOTA no se puede activar el dual con penalty = "l2")
            self.box, self, "dual", label="Dual", callback = self.selection
        )
        
        self.box.setDisabled(True)                                   # Desactivamos el panel


    @Inputs.data                        # Decorador para los datos de entrada
    def set_data(self, dataset):        # Metodo para establecer los datos
        if dataset is not None:         
            self.dataset = dataset
            self.box.setDisabled(False) 
            self.set_penalty()      
            self.selection()   
        else:
            self.dataset = None
            self.data = None
            self.Outputs.reducedData.send(self.dataset)   # Envia el conjunto seleccionado aleatoriamente por el canal sample
            self.Outputs.data.send(self.data)     # Envia el resto del conjunto por el canal other

    
    #Decidimos el tipo de norma que usamos
    def set_penalty(self):
        if self.penaltyLabel == 0:      
            self.penalty = "l1";
        else:
            self.penalty = "l2";

    def selection(self):    
        if self.dataset is None:
            return
        
        
        self.Outputs.reducedData.send(self.dataset)   # Envia el conjunto seleccionado aleatoriamente por el canal sample
        self.Outputs.data.send(self.dataset)     # Envia el resto del conjunto por el canal other