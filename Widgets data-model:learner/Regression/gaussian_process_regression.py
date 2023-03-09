import numpy as np  # importamos la librería numpy
import sklearn.gaussian_process as gaussian_process # importamos el módulo de proceso gaussiano de scikit-learn
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel    # importamos dos núcleos para el proceso gaussiano

from Orange.regression import Learner, Model, SklLearner, SklModel   # importamos clases de Orange para regresión y modelado
from Orange.preprocess import Normalize # importamos la clase de preprocesamiento de Orange para normalizar los datos
from Orange.preprocess.score import LearnerScorer   # importamos una clase de puntuación de aprendices de Orange
from Orange.data import Variable, ContinuousVariable     # importamos dos clases de Orange para definir variables

__all__ = ["gausianProcessLearner"] # Define una lista de objetos que se importan al utilizar la sintaxis 'from [módulo] import *'.


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable # definimos el tipo de variable de características como Variable en Orange
    class_type = ContinuousVariable # definimos el tipo de variable de clase como ContinuousVariable en Orange

    def score(self, data): # definimos un método para puntuar las características en función del modelo generado por el aprendiz
        data = Normalize()(data)    # normalizamos los datos de entrada
        model = self(data) # generamos un modelo utilizando el aprendiz (self) y los datos normalizados
        return np.abs(model.coefficients), model.domain.attributes  # devolvemos el valor absoluto de los coeficientes del modelo y las características utilizadas en el modelo

class GaussianProcessLearner(SklLearner, _FeatureScorerMixin):
    __wraps__ = gaussian_process.GaussianProcessRegressor   # envolvemos el proceso gaussiano de scikit-learn en una clase de aprendiz

    def __init__(self, kernel=DotProduct() + WhiteKernel(), random_state = 0, n_restarts_optimizer = 0, normalize_y = False,
                 copy_X_train = False, preprocessors=None): # definimos los parámetros del aprendiz
        super().__init__(preprocessors=preprocessors)   # inicializamos la superclase SklLearner con los preprocesadores proporcionados
        self.params = vars()    # guardamos los parámetros proporcionados al aprendiz