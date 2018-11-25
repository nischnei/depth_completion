import os

import tensorflow as tf
from experiment import Experiment

from visualization import depth_colored


class DepthCompletionExperiment(Experiment):

    def __init__(self):
        super(DepthCompletionExperiment, self).__init__()
        self.parameters.image_size = (352, 1216)

    def load_img_to_tensor(self, dict_type_to_imagepath):
        dict_res = {}
        for str_type, str_filepath in dict_type_to_imagepath.items():
            try:
                kittipath = os.environ['KITTIPATH']
                str_filepath = tf.regex_replace(str_filepath, tf.constant(
                    '\$KITTIPATH'), tf.constant(kittipath))
            except Exception:
                print("WARNING: KITTIPATH not defined - this may result in errors!")
            tf_filepath = tf.read_file(str_filepath)
            tf_tensor = tf.image.decode_png(tf_filepath, dtype=tf.uint16)
            tf_tensor = tf.image.resize_image_with_crop_or_pad(tf_tensor, self.parameters.image_size[0], self.parameters.image_size[1]) 
            tf_tensor = tf.cast(tf_tensor, dtype=tf.float32)
            tf_tensor = tf.divide(tf_tensor, 256.0)

            dict_res[str_type] = tf_tensor
        return dict_res


    def input_fn(self, dataset, mode="train"):
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
                        buffer_size=self.parameters.steps_per_epoch / self.parameters.batchsize)
                tf_dataset = tf_dataset.map(
                    self.load_img_to_tensor, num_parallel_calls=1)
                tf_dataset = tf_dataset.batch(self.parameters.batchsize)
                tf_dataset = tf_dataset.prefetch(
                    buffer_size=self.parameters.prefetch_buffer_size)
            else:
                tf_dataset = tf_dataset.map(
                    self.load_img_to_tensor, num_parallel_calls=1)
                tf_dataset = tf_dataset.batch(1)
                tf_dataset = tf_dataset.prefetch(
                    buffer_size=self.parameters.prefetch_buffer_size)

            iterator = tf_dataset.make_one_shot_iterator()

            dict_tf_input = iterator.get_next()

            tf_input = dict_tf_input["input"]
            tf_label = dict_tf_input["label"]
            
            if mode == "train":
                tf_input.set_shape([self.parameters.batchsize,
                                    self.parameters.image_size[0], self.parameters.image_size[1], 1])
                tf_input = tf.cast(tf_input, tf.float32)
                tf_label.set_shape([self.parameters.batchsize,
                                    self.parameters.image_size[0], self.parameters.image_size[1], 1])
                tf_label = tf.cast(tf_label, tf.float32)
            else:
                tf_input.set_shape([1, self.parameters.image_size[0], self.parameters.image_size[1], 1])
                tf_input = tf.cast(tf_input, tf.float32)
                tf_label.set_shape([1, self.parameters.image_size[0], self.parameters.image_size[1], 1])
                tf_label = tf.cast(tf_label, tf.float32)

        return tf_input, tf_label

    def input_fn_train(self):
        return self.input_fn(self.parameters.dataset_train)

    def input_fn_val(self):
        return self.input_fn(self.parameters.dataset_val, mode="val")

    def replaceKITTIPath(self, _string):
        try:
            kittipath = os.environ['KITTIPATH']
            _string = _string.replace('$KITTIPATH', kittipath)
        except Exception:
            print("WARNING: KITTIPATH not defined - this may result in errors!")

        return _string

    # Define the model function (following TF Estimator Template)
    def model_fn(self, features, labels, mode):
        with tf.name_scope("Model"):
            # Build the neural network
            # Because Dropout have different behavior at training and prediction time, we
            # need to create 2 distinct computation graphs that still share the same weights.
            logits_train = self.network(
                features, reuse=False, is_training=True)
            logits_test = self.network(features, reuse=True, is_training=False)

            if mode == tf.estimator.ModeKeys.TRAIN:
                tf.summary.image(
                    "Input/Train/", depth_colored(features, tf.reduce_max(labels)))
                tf.summary.image(
                    "Output/Train/", depth_colored(logits_train, tf.reduce_max(labels)))
                tf.summary.image(
                    "Label/Train/", depth_colored(labels, tf.reduce_max(labels)))

            # Predictions
            prediction = logits_test

            # If prediction mode, early return
            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode, predictions=prediction)

            # label weight mask
            label_mask = tf.where(tf.equal(labels, self.parameters.invalid_value),
                                  tf.zeros_like(labels),
                                  tf.ones_like(labels))

            norm = 1. / (self.parameters.image_size[0]*self.parameters.image_size[1])

            normed_label_mask = label_mask * norm

            if mode == tf.estimator.ModeKeys.TRAIN:
                tf.summary.image(
                    "Mask/Train/", normed_label_mask)

            with tf.name_scope("Loss") as loss_scope:
                # Define loss and optimizer
                loss = tf.reduce_mean(self.parameters.loss_function(predictions=logits_train,
                                                                    labels=labels,
                                                                    weights=normed_label_mask))
            # specify what should be done during the TRAIN call
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = self.parameters.optimizer.minimize(loss=loss,
                                                              global_step=tf.train.get_global_step())

                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

            # specify what should be done during the EVAL call
            # Evaluate the accuracy of the model (MAE)
            mae_op = tf.metrics.mean_absolute_error(
                labels=labels, predictions=prediction, weights=normed_label_mask)
            # Evaluate the accuracy of the model (MSE)
            mse_op = tf.metrics.mean_squared_error(
                labels=labels, predictions=prediction, weights=normed_label_mask)

            return tf.estimator.EstimatorSpec(
                mode=mode, predictions=prediction, loss=loss, eval_metric_ops={'mae': mae_op, 'mse': mse_op})

    def train(self):
        # Build the Estimator
        model = tf.estimator.Estimator(
            self.model_fn, model_dir=self.parameters.log_dir)

        train_spec = tf.estimator.TrainSpec(
            input_fn=self.input_fn_train, max_steps=self.parameters.num_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=self.input_fn_val)

        # Train and evaluate the Model
        tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

    def val(self):
        # Clear session and prepare for testing
        pass

    def test(self):
        input_data, label_data = self.load_data(self.parameters.dataset_test)
