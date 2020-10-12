"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2020 All Rights Reserved.
"""
import logging
import keras
import time
import json
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.backend import set_session

from keras import backend as k
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image.numpy_array_iterator import NumpyArrayIterator

from ibmfl.util import config
from ibmfl.model.fl_model import FLModel
from ibmfl.model.model_update import ModelUpdate
from ibmfl.exceptions import FLException, LocalTrainingException

logger = logging.getLogger(__name__)


class KerasFLModel(FLModel):
    """
    Wrapper class for importing keras and tensorflow.keras models.
    """

    def __init__(self, model_name, model_spec, keras_model=None):
        """
        Create a `KerasFLModel` instance from a Keras model.
        If keras_model is provided, it will use it; otherwise it will take
        the model_spec to create the model.
        Assumes the `model` passed as argument is compiled.

        :param model_name: String specifying the type of model e.g., Keras_CNN
        :type model_name: `str`
        :param model_spec: Specification of the keras_model
        :type model_spec: `dict`
        :param keras_model: Compiled keras model.
        :type keras_model: `keras.models.Model`
        """
        self.graph = tf.get_default_graph()
        self.sess = tf.Session()
        set_session(self.sess)

        if keras_model is None:
            if model_spec is None or (not isinstance(model_spec, dict)):
                raise ValueError('Initializing model requires '
                                 'a model specification or '
                                 'compiled keras model. '
                                 'None was provided')
            # In this case we need to recreate the model from model_spec
            self.model = self.load_model_from_spec(model_spec)
        else:
            if not issubclass(type(keras_model), (keras.models.Model,
                                                  tf.keras.models.Model)):
                raise ValueError('Compiled keras model needs to be provided '
                                 '(keras.models/tensorflow.keras.models). '
                                 'Type provided' + str(type(keras_model)))

            self.model = keras_model

        self.model_type = model_name
        # keras flag
        if issubclass(type(self.model), keras.models.Model):
            self.is_keras = True
        else:
            self.is_keras = False

        # Default values for local training
	############# 
        self.batch_size = 32 ##### changed this by GEC_BATCH_23
        self.epochs = 1
        self.steps_per_epoch = 100

    def fit_model(self, train_data, fit_params=None):
        """
        Fits current model with provided training data.

        :param train_data: Training data, a tuple given in the form \
        (x_train, y_train) or a datagenerator of of type `keras.utils.Sequence`, \
        `keras.preprocessing.image.ImageDataGenerator`
        :type train_data: `np.ndarray`
        :param fit_params: (optional) Dictionary with hyperparameters \
        that will be used to call Keras fit function.\
        Hyperparameter parameters should match keras expected values \
        e.g., `epochs`, which specifies the number of epochs to be run. \
        If no `epochs` or `batch_size` are provided, a default value \
        will be used (1 and 128, respectively).
        :type fit_params: `dict`
        :return: None
        """
        # Initialized with default values
        batch_size = self.batch_size
        epochs = self.epochs
        steps_per_epoch = self.steps_per_epoch
        # Extract x_train and y_train, by default,
        # label is stored in the last column

        # extract hyperparams from fit_param
        if fit_params and ('hyperparams' in fit_params):
            hyperparams = fit_params['hyperparams']
            try:
                training_hp = hyperparams['local']['training']

                if 'batch_size' in training_hp:
                    batch_size = training_hp['batch_size']
                else:
                    # In this case, use default values.
                    logger.info('Using default hyperparameters: '
                                ' batch_size:' + str(self.batch_size))

                if 'epochs' in training_hp:
                    epochs = training_hp['epochs']
                else:
                    # In this case, use default values.
                    logger.info('Using default hyperparameters: '
                                ' epochs:' + str(self.epochs))

                if 'steps_per_epoch' in training_hp:
                    steps_per_epoch = training_hp.get('steps_per_epoch')

            except Exception as ex:
                logger.exception(str(ex))
                logger.warning('Hyperparams badly formed.')
                # In this case, use default values.
                logger.info('Using default hyperparameters: '
                            'epochs:' + str(self.epochs) +
                            ' batch_size:' + str(self.batch_size))

        try:

            if type(train_data) is tuple and type(train_data[0]) is np.ndarray:
                self.fit(
                    train_data, batch_size=batch_size, epochs=epochs)

            else:
                self.fit_generator(
                    train_data, batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch)

        except Exception as e:
            logger.exception(str(e))
            if epochs is None:
                logger.exception('epochs need to be provided')

            raise LocalTrainingException(
                'Error occurred while performing model.fit')

    def fit(self, train_data, batch_size, epochs):
        """
        Fits current model using model.fit with provided training data.

        :param train_data: Training data, a tuple \
        given in the form (x_train, y_train).
        :type train_data: `np.ndarray`
        :param batch_size: Number of samples per gradient update.
        :type batch_size: Integer
        :param epochs: Number of epochs to train the model.
        :type epochs: Integer
        :return: None
        """
        x = train_data[0]
        y = train_data[1]
        with self.graph.as_default():
            set_session(self.sess)
            self.model.fit(x, y, batch_size=batch_size, epochs=epochs)

    def fit_generator(self, training_generator, batch_size, epochs, steps_per_epoch=None):
        """
        Fits current model using model.fit_generator with provided training data generator.

        :param train_data: Training datagenerator of of type `keras.utils.Sequence`, \
        `keras.preprocessing.image.ImageDataGenerator`
        :type train_data: `ImageDataGenerator` or `keras.utils.Sequence`
        :param batch_size: Number of samples per gradient update.
        :type batch_size: Integer
        :param epochs: Number of epochs to train the model.
        :type epochs: Integer
        :param steps_per_epoch: Total number of steps (batches of samples) \
                to yield from `generator` before declaring one epoch. Optional 
                for `Sequence` data generator`
                as a number of steps.
        :type steps_per_epoch: `int`
        :return: None
        """

        if type(training_generator) is NumpyArrayIterator and not steps_per_epoch:
            raise LocalTrainingException(
                "Variable steps_per_epoch cannot be None for generators not \
                    of type keras.utils.Sequence!")

        with self.graph.as_default():
            set_session(self.sess)
            self.model.fit_generator(
                training_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)

    def update_model(self, model_update):
        """
        Update keras model with provided model_update, where model_update
        should be generated according to `KerasFLModel.get_model_update()`.

        :param model_update: `ModelUpdate` object that contains the weight \
        that will be used to update the model.
        :type model_update: `ModelUpdate`
        :return: None
        """
        if isinstance(model_update, ModelUpdate):
            with self.graph.as_default():
                set_session(self.sess)
                w = model_update.get("weights")
                self.model.set_weights(w)
        else:
            raise LocalTrainingException('Provided model_update should be of '
                                         'type ModelUpdate. '
                                         'Instead they are:' +
                                         str(type(model_update)))

    def get_model_update(self):
        """
        Generates a `ModelUpdate` object that will be sent to other entities.

        :return: ModelUpdate
        :rtype: `ModelUpdate`
        """
        w = self.model.get_weights()
        return ModelUpdate(weights=w)

    def predict(self, x, batch_size=16, **kwargs):
        """
        Perform prediction for a batch of inputs. Note that for classification
        problems, it returns the resulting probabilities.

        :param x: Samples with shape as expected by the model.
        :type x: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param kwargs: Dictionary of keras-specific arguments.
        :type kwargs: `dict`

        :return: Array of predictions
        :rtype: `np.ndarray`
        """
        with self.graph.as_default():
            set_session(self.sess)
            return self.model.predict(x, batch_size=batch_size, **kwargs)

    def evaluate(self, test_dataset, **kwargs):
        """
        Evaluates the model given testing data.
        :param test_dataset: Testing data, a tuple given in the form \
        (x_test, test) or a datagenerator of of type `keras.utils.Sequence`, 
        `keras.preprocessing.image.ImageDataGenerator`
        :type test_dataset: `np.ndarray`

        :param kwargs: Dictionary of metrics available for the model
        :type kwargs: `dict`
        """

        if type(test_dataset) is tuple:
            x_test = test_dataset[0]
            y_test = test_dataset[1]

            return self.evaluate_model(x_test, y_test)

        else:
            return self.evaluate_generator_model(
                test_dataset)

    def evaluate_model(self, x, y, batch_size=16, **kwargs):
        """
        Evaluates the model given x and y.

        :param x: Samples with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Corresponding labels to x
        :type y: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param kwargs: Dictionary of metrics available for the model
        :type kwargs: `dict`
        """
        with self.graph.as_default():
            set_session(self.sess)
            metrics = self.model.evaluate(x, y, batch_size=128, **kwargs)
            names = self.model.metrics_names
            dict_metrics = {}

            if type(metrics) == list:
                for metric, name in zip(metrics, names):
                    dict_metrics[name] = metric
            else:
                dict_metrics[names[0]] = metrics
        return dict_metrics

    def evaluate_generator_model(self, test_generator, batch_size=16, **kwargs):
        """
        Evaluates the model based on the provided data generator.

        :param test_generator: Testing datagenerator of of type `keras.utils.Sequence`, \
        `keras.preprocessing.image.ImageDataGenerator`
        :type train_data: `ImageDataGenerator` or `keras.utils.Sequence`
        :param batch_size: Number of samples per gradient update.
        :type batch_size: Integer

        :return: metrics
        :rtype: `dict`
        """
        steps = self.steps_per_epoch
        if 'steps_per_epoch' in kwargs:
            steps = kwargs['steps_per_epoch']

        if not type(test_generator) is NumpyArrayIterator and not steps:
            raise LocalTrainingException(
                "Variable steps_per_epoch cannot be None for generator not of type keras.utils.Sequence")
        with self.graph.as_default():
            metrics = self.model.evaluate_generator(
                test_generator, steps=steps)
            names = self.model.metrics_names
            dict_metrics = {}

            if type(metrics) == list:
                for metric, name in zip(metrics, names):
                    dict_metrics[name] = metric
            else:
                dict_metrics[names[0]] = metrics

        return dict_metrics

    def save_model(self, filename=None):
        """
        Save a model to file in the format specific to the backend framework.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path is \
        specified, the model will be stored in the default data location of \
        the library `DATA_PATH`.
        :type path: `str`
        :return: filename
        """
        if filename is None:
            filename = 'model_{}.h5'.format(time.time())

        full_path = super().get_model_absolute_path(filename)
        self.model.save(full_path)
        logger.info('Model saved in path: %s.', full_path)
        return filename

    @staticmethod
    def load_model(file_name, custom_objects={}):
        """
        Loads a model from disk given the specified file_name

        :param file_name: Name of the file that contains the model to be loaded.
        :type file_name: `str`
        :return: Keras model loaded to memory
        :rtype: `keras.models.Model`
        """
        # try loading model from keras
        model = KerasFLModel.load_model_via_keras(file_name,
                                                  custom_objects)
        if not model:
            # try loading model from tf.keras
            model = KerasFLModel.load_model_via_tf_keras(file_name,
                                                         custom_objects)
            if model is None:
                logger.error('Loading model failed! '
                             'An acceptable compiled model should be of type '
                             '(keras.models/tensorflow.keras.models)!')
                raise FLException(
                    'Unable to load the provided compiled model!')

        return model

    @staticmethod
    def load_model_via_keras(file_name, custom_objects={}):
        """
        Loads a model from disk given the specified file_name via keras.

        :param file_name: Name of the file that contains the model to be loaded.
        :type file_name: `str`
        :return: Keras model loaded to memory
        :rtype: `keras.models.Model`
        """
        # try loading model from keras
        model = None
        try:
            model = keras.models.load_model(
                file_name, custom_objects=custom_objects)
            model._make_predict_function()
        except Exception as ex:
            logger.error(
                'Loading model via keras.models.load_model failed!')

        return model

    @staticmethod
    def load_model_via_tf_keras(file_name, custom_objects={}):
        """
        Loads a model from disk given the specified file_name via tf.keras.

        :param file_name: Name of the file that contains the model to be loaded.
        :type file_name: `str`
        :return: tf.keras model loaded to memory
        :rtype: `tf.keras.models.Model`
        """
        # try load from tf.keras
        model = None
        try:
            model = tf.keras.models.load_model(
                file_name, custom_objects=custom_objects)
            model._make_predict_function()
        except Exception as ex:
            logger.error('Loading model via tf.keras.models.load_model '
                         'failed!')

        return model

    @staticmethod
    def model_from_json_via_keras(json_file_name):
        """
        Loads a model architecture from disk via keras
        given the specified json file name.

        :param json_file_name: Name of the file that contains \
        the model architecture to be loaded.
        :type json_file_name: `str`
        :return: Keras model with only model architecture loaded to memory
        :rtype: `keras.models.Model`
        """
        # try loading model from keras
        model = None
        json_file = open(json_file_name, 'r')
        f = json_file.read()
        json_file.close()
        try:
            model = keras.models.model_from_json(f)
        except Exception as ex:
            logger.error('Loading model via '
                         'keras.models.model_from_json failed!')

        return model

    @staticmethod
    def model_from_json_via_tf_keras(json_file_name):
        """
        Loads a model architecture from disk via tf.keras
        given the specified json file name.

        :param json_file_name: Name of the file that contains \
        the model architecture to be loaded.
        :type json_file_name: `str`
        :return: tf.keras model with only model architecture loaded to memory
        :rtype: `tf.keras.models.Model`
        """
        # try loading model from keras
        model = None
        json_file = open(json_file_name, 'r')
        f = json_file.read()
        json_file.close()
        try:
            model = tf.keras.models.model_from_json(f)
        except Exception as ex:
            logger.error(
                'Loading model via tf.keras.models.model_from_json failed! ')

        return model

    @staticmethod
    def load_model_from_spec(model_spec):
        """
        Loads model from provided model_spec, where model_spec is a `dict`
        that contains two items: model_spec['model_architecture'] has a
        pointer to the file where the keras model architecture in stored
        in json format, and model_spec['model_weights'] contains
        the path where the associated weights are stored as h5.

        :return: model
        :rtype: `keras.models.Model`
        """

        if 'model_definition' in model_spec:
            model_file = model_spec['model_definition']
            model_absolute_path = config.get_absolute_path(model_file)
            custom_objects = {}
            if 'custom_objects' in model_spec:

                custom_objects_config = model_spec['custom_objects']
                for custom_object in custom_objects_config:
                    key = custom_object['key']
                    value = custom_object['value']
                    path = custom_object['path']
                    custom_objects[key] = config.get_attr_from_path(
                        path, value)

            model = KerasFLModel.load_model(model_absolute_path,
                                            custom_objects=custom_objects)
        else:
            # Load architecture from json file
            try:
                model = KerasFLModel.model_from_json_via_keras(
                    model_spec['model_architecture'])
                if not model:
                    model = KerasFLModel.model_from_json_via_tf_keras(
                        model_spec['model_architecture'])
                    if model is None:
                        logger.error(
                            'An acceptable compiled model should be of type '
                            '(keras.models/tensorflow.keras.models)!')
                        raise FLException(
                            'Unable to load the provided compiled model!')
            except Exception as ex:
                logger.error(str(ex))
                raise FLException(
                    'Unable to load the provided compiled model!')

            # Load weights from h5 file
            if 'model_weights' in model_spec:
                model.load_weights(model_spec['model_weights'])
            # model.load_weights(weights)

            # Compile model with provided parameters:
            compiled_option = model_spec['compile_model_options']
            try:
                if 'optimizer' in compiled_option:
                    optimizer = compiled_option['optimizer']
                else:
                    logger.warning('No optimizer information was provided '
                                   'in the compile_model_options, '
                                   'set keras optimizer to default: SGD')
                    optimizer = 'sgd'
                if 'loss' in compiled_option:
                    loss = compiled_option['loss']
                else:
                    logger.warning('No loss function was provided '
                                   'in the compile_model_options.'
                                   'set keras loss function to default: None')
                    loss = None
                if 'metrics' in compiled_option:
                    metrics = compiled_option['metrics']
                    metrics = [metrics] if isinstance(
                        metrics, str) else metrics
                else:
                    logger.warning('No metrics information was provided '
                                   'in the compile_model_options,'
                                   'set keras metrics to default: None')
                    metrics = None
                model.compile(optimizer=optimizer,
                              loss=loss,
                              metrics=metrics)
            except Exception as ex:
                logger.exception(str(ex))
                logger.exception('Failed to compiled keras model.')
        return model

    def expand_model_by_layer_name(self, new_dimension, layer_name="dense"):
        """
        Expand the current Keras model with provided dimension of
        the hidden layers or model weights.
        This method by default expands the dense layer of
        the current neural network.
        It can be extends to expand other layers specified by `layer_name`,
        for example, it can be use to increase the number of CNN filters or
        increase the hidden layer size inside LSTM.

        :param new_dimension: New number of dimensions for \
        the fully connected layers
        :type new_dimension: `list`
        :param layer_name: layer's name to be expanded
        :type layer_name: `str`
        :return: None
        """
        if new_dimension is None:
            raise FLException('No information is provided for '
                              'the new expanded model. '
                              'Please provide the new dimension of '
                              'the resulting expanded model.')

        model_config = json.loads(self.model.to_json())
        i = 0

        for layer in model_config['config']['layers']:
            # find the specified layers
            if 'class_name' in layer and \
                    layer['class_name'].strip().lower() == layer_name:
                layer['config']['units'] = new_dimension[i]
                i += 1
        if self.is_keras:
            new_model = keras.models.model_from_json(json.dumps(model_config))
        else:
            new_model = tf.keras.models.model_from_json(
                json.dumps(model_config))

        metrics = self.model.metrics_names
        if 'loss' in metrics:
            metrics.remove('loss')

        new_model.compile(optimizer=self.model.optimizer,
                          loss=self.model.loss,
                          metrics=metrics)
        self.model = new_model

    def get_gradient(self, train_data):
        """
        Compute the gradient with the provided dataset at the current local
        model's weights.

        :param train_data: Training data, a tuple \
        given in the form (x_train, y_train).
        :type train_data: `np.ndarray`
        :return: gradients
        :rtype: `list` of `np.ndarray`
        """
        with self.graph.as_default():
            set_session(self.sess)
            # set up symbolic variables
            try:
                grads = self.model.optimizer.get_gradients(
                    self.model.total_loss,
                    self.model.trainable_weights)
            except Exception as ex:
                logger.exception(str(ex))
                raise FLException('Error occurred when defining '
                                  'gradient expression. ')
            symb_inputs = (self.model._feed_inputs +
                           self.model._feed_targets +
                           self.model._feed_sample_weights)

            # define the symbolic function
            if self.is_keras:
                from keras import backend as k
            else:
                from tensorflow.python.keras import backend as k

            f = k.function(symb_inputs, grads)
            try:
                x, y, sample_weight = self.model._standardize_user_data(
                    train_data[0],
                    train_data[1])
            except Exception as ex:
                logger.exception(str(ex))
                raise FLException('Error occurred when feeding data samples '
                                  'to compute current gradient.')
        return f(x + y + sample_weight)

    def is_fitted(self):
        """
        Return a boolean value indicating if the model is fitted or not.
        In particular, check if the keras model has weights.
        If it has, return True; otherwise return false.

        :return: res
        :rtype: `bool`
        """
        try:
            self.model.get_weights()
        except Exception:
            return False
        return True
