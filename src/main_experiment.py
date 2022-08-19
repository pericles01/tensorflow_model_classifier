import ast
import os.path
import numpy as np
import json

class Exp:

    def __init__(self, conf):

        # ------------------------------------- Base arguments ------------------------------------- #
        self.valdata = conf.get('valdata', None)
        self.dim = conf.get('dim', 224)
        self.resize = conf.get('resize', 'center-crop')
        self.saved_model = conf.get('saved_model', None)
        self.check_data = conf.get('check_data', False)
        self.class_names = conf.get('class_names', None)
        self.commit_hash = conf.get('commit_hash', None)
        self.experiment = conf.get('experiment', None)
        self.azureml_mlflow_uri = conf.get('azureml_mlflow_uri', None)
        self.optuna_trials = conf.get('optuna_trials', 10)
        # ------------------------------------- Azureml arguments ------------------------------------- #
        self.indir = conf.get('indir', None)
        self.outdir = conf.get('outdir', None)
        self.azureml_flag = conf.get('azure_flag', False)
        self.class_names_azureml = conf.get('class_names_azureml', None)
        self.model_name = conf.get('model_name', None)

        # ------------------------------------- Model arguments ------------------------------------- #
        self.backbone = conf.get('backbone', None)
        self.head = conf.get('head', None)
        self.pooling = conf.get('pooling', None)
        self.num_classes = conf.get('num_classes', 2)
        self.num_labels = conf.get('num_labels', 0)

        # ------------------------------------- Training arguments ------------------------------------- #

        self.epochs = conf.get('epochs', 10) # optuna
        self.batch = conf.get('batch', 16) # optuna
        if self.valdata is None:
            self.validation_split = conf.get('validation_split', 0.1)
        self.lr = conf.get('lr', 0.001) # optuna
        self.lr_decay = conf.get('lr_decay', False)
        self.dropout = conf.get('dropout', 0)
        self.augment = conf.get('augment', None)
        self.label_smoothing = conf.get('label_smoothing', False)
        self.train_backbone = conf.get('train_backbone', False)
        self.kmeans = conf.get('kmeans', False)
        self.bayesian = conf.get('bayesian', None)
        self.bayesian_samples = conf.get('bayesian_samples', None)

        # ------------------------------------- Testing arguments ------------------------------------- #
        #for classification
        self.csv_eval = conf.get('csv_eval', None)
        self.class_eval = conf.get('class_eval', None)
        self.class_overlay = conf.get('class_overlay', False)
        self.class_table = conf.get('class_table', False)
        self.class_sort = conf.get('class_sort', False)
        self.temp_filter = conf.get('temp_filter', False)
        self.filter_size = conf.get('filter_size', None)
        self.state_filter = conf.get('state_filter', None)

        # ------------------------------------- Interpretability ------------------------------------- #
        self.baseline = conf.get('baseline', 'black')
        self.interpreter = conf.get('interpreter', 'ig')
        self.cmap = conf.get('cmap', 'inferno')
        self.overlay_alpha = conf.get('overlay_alpha', 0.4)

        # intantiated by _setup()
        self.class_mode = None
        self.loss = None

        self.setup()

    def setup(self):

        if self.saved_model:
            if not self.csv_eval:
                self.read_config_file()

        """Automatically set model configuration parameters based on datasets and other arguments"""
        assert self.resize in ['downscale', 'padding', 'center-crop'], \
        "Resizing method unrecognized. Please choose either downscale, padding or center-crop"

        if self.num_classes == 2:
            self.class_mode = 'binary'
            self.loss = 'binary_crossentropy'
        else:
            self.class_mode = 'categorical'
            self.loss = 'categorical_crossentropy'
        
        if 'EfficientNet' in self.backbone and self.dropout > 0:
            self.dropout = 0.2
            print("Dropout rate is fixed for EfficientNets at 0.2")

        if self.azureml_flag:
            if self.class_names_azureml == 'None':
                self.class_names = None
            else:
                self.class_names = ast.literal_eval(self.class_names_azureml)
        
        if self.class_names is not None and self.num_labels == 0:
            if len(self.class_names) != self.num_classes:
                raise ValueError("The class names do not match the number of classes")
            if len(set(self.class_names)) != len(self.class_names):
                raise ValueError("The class names are not unique")
        
        if not isinstance(self.dim, list):  # square dim
            self.dim = [self.dim, self.dim]
        
        if self.augment is None: # or len(self.augment) == 0:
            self.augment = []


    def read_config_file(self):
        """
        Read args.json file and update attributes accordingly
        :param self
        """
        with open(os.path.join(self.saved_model, 'args.json')) as json_file:
            json_content = json.load(json_file)

            #self.task = json_content["Training and model parameters"]["Task"]
            self.backbone = json_content["Training and model parameters"]["Backbone architecture"]
            self.dim = json_content["Training and model parameters"]["Input image dimensions"]
            self.resize = json_content["Training and model parameters"]["Resizing method"]
            if "Number of labels" in json_content["Training and model parameters"]:
                self.num_labels = json_content["Training and model parameters"]["Number of labels"]
            else:
                self.num_classes = json_content["Training and model parameters"]["Number of classes"]
                self.num_labels = 0
