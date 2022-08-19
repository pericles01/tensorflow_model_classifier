"""
Description
-----------
Script containing interpretation methods which generate saliency maps for classification models
"""

import os
import ssl
import ntpath
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.model_modifiers import GuidedBackpropagation
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.scores import BinaryScore
from dataset import InputNormalization
from dataset import  ResizingMethods
from dataset import read_image_test
from dataset import read_image_show
import models
from utils import misc
from utils.plot import plot_img_attributions
from utils.data import get_classification_data_paths
from main_experiment import Exp
import argparse
from azureml.core import Run

# Logger Control 0 (all messages are logged) - 3 (no messages are logged)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Avoid SSL certification
ssl._create_default_https_context = ssl._create_unverified_context


@dataclass
class InterpreterInput:
    """Data class containing input data needed for interpretation methods, including a classification model,
    target data, preprocessing method and a colormap"""

    def __init__(self, args):
        print("Loading model ...")
        classifier = models.Classification(model_name=args.backbone, dims=args.dim, lr=args.lr, loss=args.loss,
                                           train_backbone=args.train_backbone, num_classes=args.num_classes,
                                           num_labels=args.num_labels, dropout=args.dropout, bayesian=args.bayesian)
        classifier.model.load_weights(filepath=args.saved_model + "/weights.h5")
        self.classifier = classifier.model
        print("Model loaded successfully")
        self.preprocess_input = InputNormalization(args.backbone).normalization_function
        self.img_paths, self.gts, _ = get_classification_data_paths(args.indir, args=args)
        self.cmap = plt.get_cmap(args.cmap)


@dataclass
class IntegratedGradientsParameters:
    """Data class containing integrated gradients parameters"""

    def __init__(self, baseline, dim=None, cmap=None):
        """
        :param baseline:
        :param dim: image dimensions
        :param cmap: matplotlib cmap
        """
        self.baseline = baseline
        self.m_steps = 50
        self.overlay_alpha = 0.4
        self.ig_batch_size = 2
        self.dim = dim
        self.cmap = cmap


class IntegratedGradients:
    """
    Class for integrated gradients interpretation technique
    """

    def __init__(self, input_instance, ig_params):
        """
        :param input_instance: input object instance of InterpreterInput
        :param ig_params: instance of IntegratedGradientsParameters()
        """
        self.input = input_instance
        if ig_params.dim is None:
            ig_params.dim = [224, 224]
        self.preprocess_input = input_instance.preprocess_input
        if ig_params.baseline == 'mean':
            self.base_per_image = True
            self.baseline_type = 'Mean Base'
            self.dim = ig_params.dim
        else:
            self.base_per_image = False
            self.baseline, self.baseline_type = self.create_baseline(ig_params.baseline, ig_params.dim)
            self.baseline = self.preprocess_input(self.baseline)
            self.baseline = tf.convert_to_tensor(self.baseline, dtype=tf.float32)

    def interpolate_images(self, image, alphas):
        """
        Generate interpolated inputs between baseline and input.

        :param image: input image
        :param alphas: batch of steps for interpolation
        :return: batch of interpolated inputs
        """
        alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
        baseline_x = tf.expand_dims(self.baseline, axis=0)
        input_x = tf.convert_to_tensor(image)
        delta = input_x - baseline_x
        images = baseline_x + alphas_x * delta
        return images

    def compute_gradients(self, images, target_class_idx, args):
        """
        Compute gradients between model outputs and interpolated inputs.

        :param images: batch of interpolated images
        :param target_class_idx: id of class to generate gradients for
        :return: gradient for this batch of images
        """
        with tf.GradientTape() as tape:
            tape.watch(images)
            logits = self.input.classifier(images)
            if args.num_labels:
                probs = tf.nn.sigmoid(logits)[:, target_class_idx]
            else:
                if args.num_classes == 2:
                    probs = tf.nn.sigmoid(logits)
                else:
                    probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
        return tape.gradient(probs, images)

    @staticmethod
    def integral_approximation(gradients):
        """
        Integral approximation through averaging gradients.

        :param gradients: gradients computed for current image
        :return: integrated gradients
        """
        # riemann_trapezoidal
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)
        return integrated_gradients

    def calculate_integrated_gradients(self, image, target_class_idx, m_steps, batch_size, args):
        """
        calculate integrated gradients for the given image and target class

        :param image: input image to perform method on
        :param target_class_idx: id of class to generate integrated gradients for
        :param m_steps: number of steps used for interpolating between image and baseline
        :param batch_size: size of batch computation
        :return:
        """
        if self.base_per_image:
            self.baseline = np.stack([np.full((self.dim[0], self.dim[1]), tf.reduce_mean(image[:, :, 0])),  # R
                                      np.full((self.dim[0], self.dim[1]), tf.reduce_mean(image[:, :, 1])),  # G
                                      np.full((self.dim[0], self.dim[1]), tf.reduce_mean(image[:, :, 2]))],  # B
                                     axis=-1)
            self.baseline = self.preprocess_input(self.baseline)
            self.baseline = tf.convert_to_tensor(self.baseline, dtype=tf.float32)
        # 1. Generate alphas.
        alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)

        # Initialize TensorArray outside loop to collect gradients.
        gradient_batches = tf.TensorArray(tf.float32, size=m_steps + 1)

        # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
        for alpha in tf.range(0, len(alphas), batch_size):
            from_alpha = alpha
            to_alpha = tf.minimum(from_alpha + batch_size, len(alphas))
            alpha_batch = alphas[from_alpha:to_alpha]

            # 2. Generate interpolated inputs between baseline and input.
            interpolated_path_input_batch = self.interpolate_images(image=image, alphas=alpha_batch)

            # 3. Compute gradients between model outputs and interpolated inputs.
            gradient_batch = self.compute_gradients(images=interpolated_path_input_batch,
                                                    target_class_idx=target_class_idx, args= args)

            # Write batch indices and gradients to extend TensorArray.
            gradient_batches = gradient_batches.scatter(tf.range(from_alpha, to_alpha), gradient_batch)

        # Stack path gradients together row-wise into single tensor.
        gradients = gradient_batches.stack()

        # 4. Integral approximation through averaging gradients.
        gradients = self.integral_approximation(gradients=gradients)

        # 5. Scale integrated gradients with respect to input.
        gradients = (image - self.baseline) * gradients

        return gradients

    @staticmethod
    def create_baseline(baseline, dim):
        """
        generate the baseline used for integrated gradients

        :param baseline: baseline type, one of ['black', 'white', 'random']
        :param dim: dimension of the image, given as a list [height x width]
        :return:
        """
        if baseline == 'black':
            return np.zeros(shape=(dim[0], dim[1], 3)), 'Base 0'
        if baseline == 'white':
            return np.ones(shape=(dim[0], dim[1], 3)), 'Base 1'
        if baseline == 'random':
            return np.random.uniform(0.0, 1.0, (dim[0], dim[1], 3)), 'Random Base'
        raise ValueError("Baseline not recognized in the fetching function")

    def interpret(self, m_steps, overlay_alpha, batch_size, args):
        """Loop over target data, interpret and plot"""
        positive_preds = None
        input_func = np.argmax if args.num_classes > 2 else np.round
        resize_method = ResizingMethods(args.resize).resize
        for idx, _ in enumerate(self.input.img_paths):
            print(f'\r Processing image {idx + 1} / {len(self.input.img_paths)}', end="")
            image_predict = read_image_test(self.input.img_paths[idx], resize_method, args.dim,
                                                    self.preprocess_input)
            image_show = read_image_show(self.input.img_paths[idx], resize_method, args.dim)
            prediction = self.input.classifier.predict(image_predict)[0]
            image_predict = tf.convert_to_tensor(image_predict, dtype=tf.float32)

            if args.num_labels:
                positive_preds = np.squeeze(np.where(np.round(prediction))).tolist()
                if isinstance(positive_preds, int):
                    positive_preds = [positive_preds]
                attribution_pred_mask = []
                for idx_pos in positive_preds:
                    attributions_pred = self.calculate_integrated_gradients(image=image_predict,
                                                                            target_class_idx=idx_pos,
                                                                            m_steps=m_steps,
                                                                            batch_size=batch_size, args= args)
                    attribution_pred_mask.append(tf.reduce_sum(tf.nn.relu(attributions_pred), axis=-1))

            else:
                attributions_pred = self.calculate_integrated_gradients(image=image_predict,
                                                                        target_class_idx=input_func(prediction),
                                                                        m_steps=m_steps,
                                                                        batch_size=batch_size, args=args)
                attribution_pred_mask = tf.reduce_sum(tf.nn.relu(attributions_pred), axis=-1)
            if tf.reduce_sum(attribution_pred_mask):
                output_path = os.path.join(export_dir, ntpath.basename(self.input.img_paths[idx]))
                plot_img_attributions(image_show, output_path,
                                      prediction, self.input.gts[idx] if self.input.gts else None,
                                      attribution_pred_mask, self.baseline_type, overlay_alpha,
                                      labels=positive_preds if args.num_labels else None, cmap=self.input.cmap)


class TFKerasVisMethods:
    """Interpretation methods from tf_keras_vis library including GradCAM, GradCAM++ and SmoothGrad"""

    def __init__(self, input_instance, args):
        """
        :param input_instance: input object instance of InterpreterInput
        """
        self.input = input_instance
        self.interpreter = None
        self.baseline_type = None
        self.layer_name = None
        self.select_grad_method(args)

    def select_grad_method(self, args):
        """Select gradient-based sensitivity maps method"""
        if args.interpreter == 'gradcam':
            self.interpreter = Gradcam(self.input.classifier, model_modifier=GuidedBackpropagation(), clone=True)
            self.baseline_type = 'GradCAM'
            self.layer_name = self.infer_grad_cam_target_layer(self.input.classifier)
        elif args.interpreter == 'gradcam++':
            self.interpreter = GradcamPlusPlus(self.input.classifier, model_modifier=GuidedBackpropagation(),
                                               clone=True)
            self.baseline_type = 'GradCAM++'
            self.layer_name = self.infer_grad_cam_target_layer(self.input.classifier)
        elif args.interpreter == 'smoothgrad':
            self.interpreter = Saliency(self.input.classifier, model_modifier=ReplaceToLinear(), clone=True)
            self.baseline_type = 'SmoothGrad'
        else:
            raise ValueError("Interpretation method not recognized")

    def interpret(self, args):
        """Loop over target data, interpret and plot"""
        resize_method = ResizingMethods(args.resize).resize
        if args.num_labels:
            for idx, _ in enumerate(self.input.img_paths):
                print(f'\r Processing image {idx + 1} / {len(self.input.img_paths)}', end="")
                image_predict = read_image_test(self.input.img_paths[idx], resize_method, args.dim,
                                                        self.input.preprocess_input)
                image_show = read_image_show(self.input.img_paths[idx], resize_method, args.dim)
                prediction = self.input.classifier.predict(image_predict)[0]
                positive_preds = np.squeeze(np.where(np.round(prediction))).tolist()
                if isinstance(positive_preds, int):
                    positive_preds = [positive_preds]
                grids = []
                for idx_pos in positive_preds:
                    if args.interpreter == 'smoothgrad':
                        saliency = self.interpreter(CategoricalScore(idx_pos), image_predict,
                                                    # gradient_modifier=lambda grads: tf.nn.relu(grads),
                                                    smooth_samples=50,
                                                    # The number of calculating gradients iterations.
                                                    smooth_noise=0.20)  # noise spread level.
                        grids.append(np.uint8(saliency * 255))
                    else:
                        cam = self.interpreter(CategoricalScore(idx_pos), image_predict,
                                               penultimate_layer=self.layer_name, seek_penultimate_conv_layer=False)
                        grids.append(np.uint8(self.input.cmap(cam)[..., :3] * 255))
                if grids:
                    output_path = os.path.join(export_dir, ntpath.basename(self.input.img_paths[idx]))
                    plot_img_attributions(image_show, output_path, prediction,
                                          self.input.gts[idx] if self.input.gts else None, grids,
                                          baseline_type=self.baseline_type, overlay_alpha=args.overlay_alpha,
                                          cmap=self.input.cmap, labels=positive_preds, clip=False)
        else:
            for idx, _ in enumerate(self.input.img_paths):
                print(f'\r Processing image {idx + 1} / {len(self.input.img_paths)}', end="")
                image_predict = read_image_test(self.input.img_paths[idx], resize_method, args.dim,
                                                        self.input.preprocess_input)
                image_show = read_image_show(self.input.img_paths[idx], resize_method, args.dim)
                prediction = self.input.classifier.predict(image_predict)[0]
                if args.interpreter == 'smoothgrad':
                    saliency = self.interpreter(BinaryScore(np.round(prediction)) if args.num_classes == 2
                                                else CategoricalScore(int(np.argmax(prediction))), image_predict,
                                                # gradient_modifier=lambda grads: tf.nn.relu(grads),
                                                smooth_samples=50,  # The number of calculating gradients iterations.
                                                smooth_noise=0.20)  # noise spread level.
                    heatmap = np.uint8(saliency * 255)
                else:
                    cam = self.interpreter(BinaryScore(np.round(prediction)) if args.num_classes == 2 else
                                           CategoricalScore(int(np.argmax(prediction))), image_predict,
                                           penultimate_layer=self.layer_name, seek_penultimate_conv_layer=False)
                    heatmap = np.uint8(self.input.cmap(cam)[..., :3] * 255)
                output_path = os.path.join(export_dir, ntpath.basename(self.input.img_paths[idx]))
                plot_img_attributions(image_show, output_path, prediction,
                                      self.input.gts[idx] if self.input.gts else None,
                                      heatmap, cmap=self.input.cmap, baseline_type=self.baseline_type,
                                      overlay_alpha=args.overlay_alpha, clip=False)

    @staticmethod
    def infer_grad_cam_target_layer(model):
        """
        Code from tf_explain.core.grad_cam.GradCAM
        Search for the last convolutional layer to perform Grad CAM, as stated
        in the original paper.

        :param model: tf.keras model to inspect
        :returns: Name of the target layer
        """
        for layer in reversed(model.layers):
            # Select closest 4D layer to the end of the network.
            if len(layer.output_shape) == 4:
                return layer.name

        raise ValueError("Model does not seem to contain 4D layer. Grad CAM cannot be applied.")

def interpret(conf, local=False):
    args = Exp(conf)
    if not local:
        azure_tags(args)

    interpreter_input = InterpreterInput(args)

    if args.interpreter == "ig":
        params = IntegratedGradientsParameters(baseline=args.baseline, dim=args.dim, cmap=interpreter_input.cmap)
        interpreter = IntegratedGradients(interpreter_input, params)
        interpreter.interpret(params.m_steps, params.overlay_alpha, params.ig_batch_size, args)

    else:
        interpreter = TFKerasVisMethods(interpreter_input, args)
        interpreter.interpret(args)

    print("\nEnd: " + str(datetime.now()))
    print("Results saved in " + export_dir)

def azure_tags(args):
    # ------------------------------------- Base arguments ------------------------------------- #
    run.tag('task', "Interpretation")
    if args.dim:
         run.tag('dim', args.dim)
    if args.resize:
        run.tag('resize', args.resize)
    if args.check_data:
        run.tag('check_data', args.check_data)
    if args.class_names:
        run.tag('class_names', args.class_names)
    # ------------------------------------- Model arguments ------------------------------------- #
    if args.backbone:
        run.tag('backbone', args.backbone)
    if args.num_classes:
        run.tag('num_classes', args.num_classes)
    if args.num_labels:
        run.tag('num_labels', args.num_labels)
    # ------------------------------------- Interpretability ------------------------------------- #
    if args.baseline:
        run.tag('baseline', args.baseline)
    if args.interpreter:
        run.tag('interpreter', args.interpreter)
    if args.cmap:
        run.tag('cmap', args.cmap)
    if args.overlay_alpha:
        run.tag('overlay_alpha', args.overlay_alpha)

if __name__ == '__main__':
    # Need only these to be parsed because they are dynamically created by azureml SDK
    # All options are defined in the yaml file passed with --config
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, help="input data directory")
    parser.add_argument('--outdir', type=str, help="output data directory")
    parser.add_argument('--saved_model', type=str, help="saved model directory")
    parser.add_argument('--ckpt', type=str, default=None, help="path to model weight file (for pretrained weights)")
    parser.add_argument('--config', type=str, default='run_config.yml', help="path to config file of the experiment")
    args = parser.parse_args()
    
    from experiment_config import ExperimentConfig
    conf = ExperimentConfig(args.config)
    conf.add(args)

    result_path = os.path.join(args.outdir, "results")
    if not os.path.exists(result_path):
            os.makedirs(result_path, exist_ok=True)

    timeStamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    export_dir = os.path.join(result_path, timeStamp)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)

    print("Start: " + str(datetime.now()))

    run = Run.get_context()
    interpret(conf)
