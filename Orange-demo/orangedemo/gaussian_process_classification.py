import numpy as np  # importamos la librería numpy
import sklearn.gaussian_process as gaussian_process # importamos el módulo de proceso gaussiano de scikit-learn
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel    # importamos dos núcleos para el proceso gaussiano

from Orange.classification import SklLearner  # importamos la clase SklLearner de Orange para contruir y entrenar modelo de clasificacion
from Orange.preprocess import Normalize # importamos la clase de preprocesamiento de Orange para normalizar los datos
from Orange.preprocess.score import LearnerScorer   # importamos una clase de puntuación de aprendices de Orange
from Orange.data import Variable, DiscreteVariable # importamos dos clases de Orange para definir variables

class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable         #Representa los datos de entrada, se inicializa a Variable para que no se excluya ningún tipo de dato
    class_type = DiscreteVariable   #Representa la clase de salida, con la que se realizarán las predicciones.

    def score(self, data):  # definimos un método para puntuar las características en función del modelo generado por el aprendiz
        data = Normalize()(data)    # normalizamos los datos de entrada
        model = self(data)  # generamos un modelo utilizando el aprendiz (self) y los datos normalizados
        return np.abs(model.coefficients), model.domain.attributes  # devolvemos el valor absoluto de los coeficientes del modelo y las características utilizadas en el modelo

class GaussianProcessLearner(SklLearner, _FeatureScorerMixin):
    __wraps__ = gaussian_process.GaussianProcessClassifier  # envolvemos el proceso gaussiano de scikit-learn en una clase de aprendiz
    preprocessors = SklLearner.preprocessors    #Definimos un conjunto de preprocesos

    def __init__(self, kernel=DotProduct() + WhiteKernel(), random_state = 0, n_restarts_optimizer = 0, warm_start = False,
                 copy_X_train = False,max_iter_predict = 100, preprocessors=None):  # definimos los parámetros del aprendiz
        super().__init__(preprocessors=preprocessors)   # inicializamos la superclase SklLearner con los preprocesadores proporcionados
        self.params = vars()    # guardamos los parámetros proporcionados al aprendiz