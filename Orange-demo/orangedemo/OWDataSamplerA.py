import numpy    # Importamos la librería numpy
import Orange.data  # Importamos los datos de Orange
from orangewidget.widget import OWBaseWidget, Input, Output # Importamos la clase base, la de entrada y salida
from orangewidget.settings import Setting   # Importamos setting para configurar los parametros por defecto
from orangewidget import gui    # Importamos gui para construir el panel de configuración

class OWDataSamplerA(OWBaseWidget):
    name = "Data Sampler"       # Nombre del widget
    description = "Randomly selects a subset of instances from the dataset" # Descripcion del widget
    icon = "icons/DataSamplerA.svg" # Icono del widget
    priority = 10   # Prioridad de carga del widget

    class Inputs:           # Definimos la clase de entrada con el dato que recibe el widget
        data = Input("Data", Orange.data.Table)

    class Outputs:          # Definimos la clase de salida con los datos que devuelve el widget
        sample = Output("Sampled Data", Orange.data.Table)
        other = Output("Other Data", Orange.data.Table)

    proportion = Setting(50)        # Establecemos por defecto el porcentaje de filas aleatorias al 50%
    commitOnChange = Setting(0)     # Establecemos por defecto como false los commit automaticos

    #want_main_area = True

    def __init__(self):
        super().__init__()

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")       # Creamos una caja en el widget
        # Introducimos las siguientes etiquetas en la caja
        self.infoa = gui.widgetLabel(box, 'No data on input yet, waiting to get something.')
        self.infob = gui.widgetLabel(box, '')
        self.infoc = gui.widgetLabel(box, '')

        gui.separator(self.controlArea) # Ponemos un separador
        self.optionsBox = gui.widgetBox(self.controlArea, "Options")    # Creamos otra caja
        gui.spin(self.optionsBox, self, 'proportion',                   # Agregamos un spinBox para elegir el porcentaje de filas aleatoreas
                 minv=10, maxv=90, step=10, label='Sample Size [%]:',
                 callback=[self.selection, self.checkCommit])
        gui.checkBox(self.optionsBox, self, 'commitOnChange',           # Agregamos el checkBox para seleccionar si los commit se hacen automaticamente o los hacemos a mano
                     'Commit data on selection change')
        gui.button(self.optionsBox, self, "Commit", callback=self.commit)   # Agregamos un boton
        self.optionsBox.setDisabled(True)                                   # Desactivamos el panel


    @Inputs.data                        # Decorador para los datos de entrada
    def set_data(self, dataset):        # Metodo para establecer los datos
        if dataset is not None:         
            self.dataset = dataset
            self.infoa.setText('%d instances in input dataset' % len(dataset))
            self.optionsBox.setDisabled(False)
            self.info.set_input_summary("Original dataset: " + str(len(self.dataset)))      
            self.selection()
        else:
            self.dataset = None
            self.sample = None
            self.other = None
            self.infoa.setText('No data on input yet, waiting to get something.')
            self.infob.setText('')
        self.commit()

    def selection(self):    
        if self.dataset is None:
            return

        n_selected = int(numpy.ceil(len(self.dataset) * self.proportion / 100.))    # Calcula el número de elementos que se deben incluir en la muestra
        indices = numpy.random.permutation(len(self.dataset))   # Permutar aleatoriamente los índices del conjunto de datos
        indices_sample = indices[:n_selected]       # Selecciona los primeros n_selected índices como muestra
        indices_other = indices[n_selected:]    # Selecciona los índices restantes como otra parte de los datos
        self.sample = self.dataset[indices_sample]   # Crea una muestra de los elementos del conjunto de datos seleccionados
        self.other = self.dataset[indices_other]    # Crea otra parte de los elementos del conjunto de datos seleccionados
        self.info.set_output_summary("Dataset sample:" + str(len(self.sample)) + "\nDataset other:" + str(len(self.other))) # Establecer el resumen de salida de la muestra y otra parte de los datos
        self.infob.setText('%d sampled instances' % len(self.sample))   # Establecer el texto de salida para la muestra
        self.infoc.setText('%d other instances' % len(self.other))   # Establecer el texto de salida para otra parte de los datos

    def commit(self):
        self.Outputs.sample.send(self.sample)   # Envia el conjunto seleccionado aleatoriamente por el canal sample
        self.Outputs.other.send(self.other)     # Envia el resto del conjunto por el canal other
        
    def checkCommit(self):
        if self.commitOnChange:     # Si se habilita el comiteo automatico, se llama a la función anterior
            self.commit()