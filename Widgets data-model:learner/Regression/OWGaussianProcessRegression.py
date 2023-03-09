from itertools import chain # importamos la función chain del módulo itertools
import numpy as np  # importamos la librería numpy
from AnyQt.QtCore import Qt # importamos la constante Qt desde el módulo 

from Orange.data import Table, Domain, ContinuousVariable, StringVariable  # importamos las clases Table, Domain, ContinuousVariable y StringVariable del módulo Orange.data
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel     # importamos los kernel (funciones de covarianza)
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import PairwiseKernel
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RationalQuadratic
from orangedemo.gaussian_process_regression import GaussianProcessLearner   # importamos la clase GaussianProcessLearner del módulo orangedemo.gaussian_process_regression
from Orange.widgets import settings, gui    # importamos las clases settings y gui del módulo Orange.widgets
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner  # importamos la clase OWBaseLearner del módulo Orange.widgets.utils.owlearnerwidget
from Orange.widgets.utils.signals import Output # importamos la clase Output del módulo Orange.widgets.utils.signals
from Orange.widgets.utils.widgetpreview import WidgetPreview    # importamos la clase WidgetPreview del módulo Orange.widgets.utils.widgetpreview


class OWGaussianProcess(OWBaseLearner):
    name = "Gaussian Process Regression"     # nombre del widget
    description = "The Gaussian Process regression algorithm"   # descripción del widget
    icon = "GP.png" # icono del widget
    priority = 60   # prioridad del widget

    LEARNER = GaussianProcessLearner    # clase que utilizaremos para el entrenamiento
    
    #Definición de los parametros por defecto
    kernelLabel = "DotProduct + WhiteKernel"
    kernel = DotProduct() + WhiteKernel()
    copy_X_train = settings.Setting(False)
    n_restarts_optimizer = settings.Setting(0)
    normalize_y = settings.Setting(False)
    max_iter_predict = settings.Setting(100)
    random_state = settings.Setting(0)

    kernel_types=["DotProduct + WhiteKernel", "matern kernel", "RBF", "Rational Quadratic"] #Definimos el vector con los distintos tipos de kernel
    
    def add_main_layout(self):
        box = gui.widgetBox(self.controlArea, box=True) # creamos una caja en el widget
        self.kernel_combo = gui.comboBox(   # añadimos un comboBox para seleccionar el kernel que se utilizará
            box, self, "kernelLabel", label="Kernel type: ", items=self.kernel_types, orientation=Qt.Horizontal,
            callback=self.set_kernel
        )
        
        self.copy = gui.checkBox(    # Agregamos el comboBox para seleccionar el tipo de kernel
            box, self, "copy_X_train", label="Persistent copy of the training data",  callback=self.settings_changed
        )
        
        self.normalize = gui.checkBox(   # Agregamos el checkBox para seleccionar si se normaliza o no la variable objetivo
            box, self, "normalize_y", label="Normalize y",  callback=self.settings_changed
        )
        
        self.restarts = gui.spin(   # Agregamos un spinBox para elegir el número de reinicios del optimizador
            box, self, "n_restarts_optimizer", minv=0, maxv=5, step=1, label="n_restarts_optimizer",  callback=self.settings_changed
        )
        
        self.max = gui.spin(    # Agregamos un spinBox para elegir el número máximo de iteraciones en la fase de predicción
            box, self, "max_iter_predict", minv=10, maxv=1000, step=10, label="max_iter_predict",  callback=self.settings_changed
        )
        
        self.random = gui.spin( # Spin box para seleccionar el estado aleatorio del generador de números aleatorios
            box, self, "random_state", minv=0, maxv=10, step=1, label="random_state",  callback=self.settings_changed
        )
        
        
    def set_kernel(self):
        # Dependiendo de la opción seleccionada en el combo box, se cambia el kernel
        if self.kernelLabel == 0:
            self.kernel = DotProduct() + WhiteKernel()
        elif self.kernelLabel == 1:
            self.kernel = Matern()
        elif self.kernelLabel == 2:
            self.kernel = RBF()
        elif self.kernelLabel == 3:
            self.kernel = RationalQuadratic()
        self.settings_changed()
        

    def create_learner(self):
        # Crea un objeto GaussianProcessLearner con los parámetros seleccionados en la interfaz
        return self.LEARNER(
            kernel=self.kernel,
            random_state=self.random_state,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y = self.normalize_y,
            copy_X_train=self.copy_X_train,
            preprocessors=self.preprocessors,
        )

    def update_model(self):
        # Actualiza el modelo de la misma forma que en la clase base, OWBaseLearner
        super().update_model()

if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWLogisticRegression).run(Table("zoo"))
