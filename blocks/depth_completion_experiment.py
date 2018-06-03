import os

from keras.layers import Input
from keras.models import Model
from keras import backend as K
import tensorflow as tf
from experiment import Experiment

def load_img_to_tensor(dict_type_to_imagepath):
    dict_res = {}
    for str_type, str_filepath in dict_type_to_imagepath.items():
        try:
            kittipath = os.environ['KITTIPATH']
            str_filepath = tf.regex_replace(str_filepath, tf.constant('\$KITTIPATH'), tf.constant(kittipath))
        except Exception:
            print("WARNING: KITTIPATH not defined - this may result in errors!")
        # str_filepath = tf.Print(str_filepath,[str_filepath])
        tf_filepath = tf.read_file(str_filepath)
        tf_tensor = tf.image.decode_png(tf_filepath, dtype=tf.uint16)
        tf_tensor = tf.cast(tf_tensor, dtype=tf.int32)

        dict_res[str_type] = tf_tensor
    return dict_res


class DepthCompletionExperiment(Experiment):

    def __init__(self):
        super(DepthCompletionExperiment, self).__init__()

        self.parameters.image_size = (1216, 352)
        self.parameters.loss_function = 'mean_squared_error'
        self.parameters.metrics = ['accuracy']

    def tf_data_api(self, dataset, mode="train"):

        self.dict_dataset_lists = {}
        ds_input = os.path.expandvars(dataset["input"])
        ds_label = os.path.expandvars(dataset["label"])
        self.dict_dataset_lists["input"] = tf.data.TextLineDataset(ds_input)
        self.dict_dataset_lists["label"] = tf.data.TextLineDataset(ds_label)

        with tf.name_scope("Dataset_API"):
            tf_dataset = tf.data.Dataset.zip(self.dict_dataset_lists)

            if mode == "train":
                tf_dataset = tf_dataset.repeat(self.parameters.max_epochs)
                if self.parameters.shuffle:
                    tf_dataset = tf_dataset.shuffle(
                        buffer_size=self.parameters.steps_per_epoch * self.parameters.batchsize)
                tf_dataset = tf_dataset.map(
                    load_img_to_tensor, num_parallel_calls=1)
                tf_dataset = tf_dataset.batch(self.parameters.batchsize)
            else:
                tf_dataset = tf_dataset.batch(1)

            iterator = tf_dataset.make_one_shot_iterator()

            dict_tf_input = iterator.get_next()

            tf_input = dict_tf_input["input"]
            tf_label = dict_tf_input["label"]

            tf_input = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(
                img, self.parameters.image_size[0], self.parameters.image_size[1]), tf_input)
            tf_label = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(
                img, self.parameters.image_size[0], self.parameters.image_size[1]), tf_label)

            tf_input.set_shape([self.parameters.batchsize,
                                self.parameters.image_size[0], self.parameters.image_size[1], 1])
            tf_input = tf.cast(tf_input, tf.float32)
            tf_label.set_shape([self.parameters.batchsize,
                                self.parameters.image_size[0], self.parameters.image_size[1], 1])
            tf_label = tf.cast(tf_label, tf.float32)

        return tf_input, tf_label

    def load_data(self, dataset):
        return self.tf_data_api(dataset)

    def train(self):
        tf_inputs, tf_labels = self.load_data(self.parameters.dataset_train)
        inputs = Input(tensor=tf_inputs)
        outputs = self.network(inputs)
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer=self.parameters.optimizer,
                      loss=self.parameters.loss_function,
                      metrics=self.parameters.metrics,
                      target_tensors=[tf_labels])
        model.summary()

        model.fit(epochs=self.parameters.max_epochs,
                  steps_per_epoch=self.parameters.steps_per_epoch)

        # ##
        model.save_weights('trained_model.h5')

    def val(self):

        # Clear session and prepare for testing
        K.clear_session()
        K.set_learning_phase(0)

        inputs, labels = self.load_data(self.parameters.dataset_val)
        outputs = self.network(inputs)
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer=self.parameters.optimizer,
                      loss=self.parameters.loss_function,
                      metrics=self.parameters.metrics)

        model.load_weights('trained_model.h5')

        loss, acc = model.evaluate(inputs,
                                   labels,
                                   verbose=1,
                                   batch_size=1)

        print('\nTest accuracy: {0}'.format(acc))

    def test(self):
        input_data, label_data = self.load_data(self.parameters.dataset_test)
