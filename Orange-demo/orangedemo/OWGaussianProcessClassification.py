#Importamos los tipos de kernel
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import PairwiseKernel
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RationalQuadratic
from orangedemo.gaussian_process_classification import GaussianProcessLearner   # importamos la clase GaussianProcessLearner del módulo orangedemo.gaussian_process_classification
from Orange.widgets import settings, gui     #importamos las clases settings y gui del módulo Orange.widgets
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner  # importamos la clase OWBaseLearner del módulo Orange.widgets.utils.owlearnerwidget



class OWGaussianProcess(OWBaseLearner):
    name = "Gaussian Process Classification"    # nombre del widget
    description = "The Gaussian Process classification algorithm"   # descripción del widget
    icon = "icons/GP.png"   # icono del widget
    priority = 60   # prioridad del widget

    LEARNER = GaussianProcessLearner     # clase que utilizaremos para el entrenamiento
    
    #Definición de los parametros por defecto
    kernelLabel = "DotProduct + WhiteKernel"
    kernel = DotProduct() + WhiteKernel()
    copy_X_train = settings.Setting(False)
    n_restarts_optimizer = settings.Setting(0)
    warm_start = settings.Setting(False)
    max_iter_predict = settings.Setting(100)
    random_state = settings.Setting(0)

    kernel_types=["DotProduct + WhiteKernel", "matern kernel", "pairwiseKernel", "RBF", "Rational Quadratic"]  #Definimos el vector con los distintos tipos de kernel
    
    def add_main_layout(self):
        box = gui.widgetBox(self.controlArea, box=True) # creamos una caja en el widget
        self.kernel_combo = gui.comboBox(       # añadimos un comboBox para seleccionar el kernel que se utilizará
            box, self, "kernelLabel", label="Kernel type: ", items=self.kernel_types,
            callback=self.set_kernel
        )
        
        self.copy = gui.checkBox(   # Agregamos el checkBox para seleccionar si queremos hacer persistente los datos de entrenamiento
            box, self, "copy_X_train", label="Persistent copy of the training data",  callback=self.settings_changed
        )
        
        self.warm = gui.checkBox(   # Agregamos el checkBox para seleccionar si reutilizamos los parámetros del modelo anterior
            box, self, "warm_start", label="Warm start",  callback=self.settings_changed
        )
        
        self.restarts = gui.spin(   # Agregamos un spinBox para elegir el número de reinicios del optimizador
            box, self, "n_restarts_optimizer", minv=0, maxv=10, step=1, label="n_restarts_optimizer",  callback=self.settings_changed
        )
        
        self.max = gui.spin(    # Agregamos un spinBox para elegir el número máximo de iteraciones en la fase de predicción
            box, self, "max_iter_predict", minv=10, maxv=1000, step=10, label="max_iter_predict",  callback=self.settings_changed
        )
        
        self.random = gui.spin(  # Spin box para seleccionar el estado aleatorio del generador de números aleatorios
            box, self, "random_state", minv=0, maxv=10, step=1, label="random_state",  callback=self.settings_changed
        )
        
    def set_kernel(self):
        # Dependiendo de la opción seleccionada en el combo box, se cambia el kernel
        if self.kernelLabel == 0:
            self.kernel = DotProduct() + WhiteKernel()
        elif self.kernelLabel == 1:
            self.kernel = Matern()
        elif self.kernelLabel == 2:
            self.kernel = PairwiseKernel()
        elif self.kernelLabel == 3:
            self.kernel = RBF()
        elif self.kernelLabel == 4:
            self.kernel = RationalQuadratic()
        self.settings_changed()
        

    def create_learner(self):
        # Crea un objeto GaussianProcessLearner con los parámetros seleccionados en la interfaz
        return self.LEARNER(
            kernel=self.kernel,
            random_state=self.random_state,
            n_restarts_optimizer=self.n_restarts_optimizer,
            warm_start = self.warm_start,
            copy_X_train=self.copy_X_train,
            max_iter_predict = self.max_iter_predict,
            preprocessors=self.preprocessors,
        )

    def update_model(self):
        # Actualiza el modelo de la misma forma que en la clase base, OWBaseLearner
        super().update_model()

