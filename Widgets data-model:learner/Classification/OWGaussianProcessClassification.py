from itertools import chain
import numpy as np
from AnyQt.QtCore import Qt

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import PairwiseKernel
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RationalQuadratic
from orangedemo.gaussian_process_classification import GaussianProcessLearner
from Orange.widgets import settings, gui
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.signals import Output
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Msg


class OWGaussianProcess(OWBaseLearner):
    name = "Gaussian Process Classification"
    description = "The Gaussian Process classification algorithm" 
    icon = "GP.png"
    priority = 60
    keywords = []

    LEARNER = GaussianProcessLearner
    
    kernelLabel = "DotProduct + WhiteKernel"
    kernel = DotProduct() + WhiteKernel()
    copy_X_train = settings.Setting(False)
    n_restarts_optimizer = settings.Setting(0)
    warm_start = settings.Setting(False)
    max_iter_predict = settings.Setting(100)
    random_state = settings.Setting(0)

    kernel_types=["DotProduct + WhiteKernel", "matern kernel", "pairwiseKernel", "RBF", "Rational Quadratic"]
    
    def add_main_layout(self):
        box = gui.widgetBox(self.controlArea, box=True)
        self.kernel_combo = gui.comboBox(
            box, self, "kernelLabel", label="Kernel type: ", items=self.kernel_types, orientation=Qt.Horizontal,
            callback=self.set_kernel
        )
        
        self.copy = gui.checkBox(
            box, self, "copy_X_train", label="Persistent copy of the training data",  callback=self.settings_changed
        )
        
        self.warm = gui.checkBox(
            box, self, "warm_start", label="Warm start",  callback=self.settings_changed
        )
        
        self.restarts = gui.spin(
            box, self, "n_restarts_optimizer", minv=0, maxv=5, step=1, label="n_restarts_optimizer",  callback=self.settings_changed
        )
        
        self.max = gui.spin(
            box, self, "max_iter_predict", minv=10, maxv=1000, step=10, label="max_iter_predict",  callback=self.settings_changed
        )
        
        self.random = gui.spin(
            box, self, "random_state", minv=0, maxv=10, step=1, label="random_state",  callback=self.settings_changed
        )
        
    def set_kernel(self):
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
        super().update_model()

if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWLogisticRegression).run(Table("zoo"))
