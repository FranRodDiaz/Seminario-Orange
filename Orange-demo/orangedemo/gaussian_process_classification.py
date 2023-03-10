import numpy as np
import sklearn.gaussian_process as gaussian_process
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from Orange.classification import SklLearner, SklModel
from Orange.preprocess import Normalize
from Orange.preprocess.score import LearnerScorer
from Orange.data import Variable, DiscreteVariable

__all__ = ["gausianProcessLearner"]


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable         #Representa los datos de entrada, se inicializa a Variable para que no se excluya ningún tipo de dato
    class_type = DiscreteVariable   #Representa la clase de salida, con la que se realizarán las predicciones.

    def score(self, data):
        data = Normalize()(data)
        model = self(data)
        return np.abs(model.coefficients), model.domain.attributes

class GaussianProcessLearner(SklLearner, _FeatureScorerMixin):
    __wraps__ = gaussian_process.GaussianProcessClassifier
    preprocessors = SklLearner.preprocessors

    def __init__(self, kernel=DotProduct() + WhiteKernel(), random_state = 0, n_restarts_optimizer = 0, warm_start = False,
                 copy_X_train = False,max_iter_predict = 100, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()