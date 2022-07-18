"""Description
   -----------
This script is used to submit training jobs to the azureml cloud"""

from __future__ import annotations

import os
from azureml.core import Workspace, Environment, Dataset, Experiment, Datastore, ComputeTarget, ScriptRunConfig, runconfig, ContainerRegistry
from azureml.data import OutputFileDatasetConfig
#from azureml.core.conda_dependencies import CondaDependencies
from azureml.exceptions import UserErrorException
from azureml.core.model import Model
from shutil import copytree, copyfile, rmtree

class AMLWDeployment:

    def __init__(self, target, config):

        self._config = config
        self._target = target
        self._output_path = self._config.get('output') +'/{run-id}'
        self._snapshot_folder = 'snapshot'

        # create by _setup()
        self._workspace = None
        self._ds_data = None
        self._ds_val_data = None
        self._ds_model_ckpt = None
        self._ds_model = None
        self._ds_output = None
        self._exp = None
        self._setup()

    def _setup(self):
        """Setup all azureml instances necessary for deployment (ScriptRunConfig)
        """

        self._workspace = self._get_workspace()

        if self._is_local_run():
            self._env = Environment('user-managed-env')
            self._env.python.user_managed_dependencies = True
            print(' * Using local python environment.')
        else:
            self._env = self._dockerimage_environment()
            

        # TODO: Need to specify version for this!
        self._ds_data = self._input_dataset()
        self._ds_val_data = self._input_val_dataset()
        self._ds_model_ckpt = self._model_ckpt()
        self._ds_model = self._input_model()
        self._ds_output = self._output_ds()
        self._compute_target = self._set_compute_target()
        self._experiment = self._set_experiment()
        self._create_snapshot()

    def _is_local_run(self):

        if self._target == 'local':
            return True

        return False

    def _get_workspace(self):

        if self._config is None:
            return None

        try:
            print('Instantiating Workspace: ' + self._config.get('workspace'))
            ws = Workspace(
                subscription_id=self._config.get('subscription'),
                resource_group=self._config.get('resource_group'),
                workspace_name=self._config.get('workspace')
            )
            ws.write_config()
            print(' * Workspace configuration succeeded')
            return ws
        except:
            print(' ! Error: Workspace not found')
            return None

    # TODO: Needs refactoring
    #def pip_environment(self, env_name, req_path='./requirements.txt') -> AzureAccess:
    #    self.__env = Environment.from_pip_requirements(env_name, req_path)
    #    self.__env.register(self.__ws)
    #    print('Environment created')
    #    return self

    #def _dockerfile_environment(self, env_name, dockerfile_name) -> AzureAccess:
    #    try:
    #       env = Environment.from_dockerfile(env_name, dockerfile_name)
    #       env.register(self.__ws)
    #       print('Environment created')
    #    except UserErrorException:
    #       print(' ! Error: Environment-file not found')
    #    return env

    # TODO: Needs refactoring
    #def conda_environment(self, env_name, existing_env_name, packages = []) -> AzureAccess:
    #    env = Environment.get(self.__ws, existing_env_name)
    #    self.__env = env.clone(env_name)
    #    conda_dep = CondaDependencies.create(pip_packages=packages)
    #    self.__env.python.conda_dependencies = conda_dep
    #    self.__env.register(self.__ws)
    #    print('Environment created')
    #    return self
    
    def _dockerimage_environment(self):
        """Create/reuse environment based on a docker image from a container registry.
        """
        try:
            container_registry = ContainerRegistry()
            env = Environment.from_docker_image(self._config.get('environment'),
                                                self._config.get('docker_image'),
                                                self._config.get('container_registry'))
        
            #env.register(self._workspace)
            print(' * Successfully instantiated docker image environment')
        except UserErrorException:
            print(' ! Environment not found')
            return None
        return env

    
    #def conda_environment(self, env_name="classifier", yaml_file_path='./environment_tf2.4.yml') -> AzureAccess:
    #    try:
    #       env = Environment.from_conda_specification(name = env_name,
    #                                         file_path = yaml_file_path)
    #       env.register(self._workspace)
    #       print('Environment created')
    #    except UserErrorException:
    #        print(' ! Error: Environment-file not found')
    #    return env


    def _input_ds_by_name(self):

        try:
            ds = Dataset.get_by_name(self._workspace, self._config.get('dataset'))
            print(' * Successfully got dataset ' + self._config.get('dataset'))
        except UserErrorException:
            print(' ! Error: Dataset not found')
            return None

        return ds
    
    def _input_dataset(self) -> AzureAccess:
        try:
            datastore = Datastore.get(self._workspace, self._config.get('datastore'))

            ds_data = Dataset.File.from_files((datastore, self._config.get('dataset')))
            print(' * Successfully created data input ' + self._config.get('dataset'))

        except UserErrorException:
            print(' ! ERROR:Dataset not found')
            return None

        return ds_data 
    
    def _input_val_dataset(self):
        try:
            datastore = Datastore.get(self._workspace, self._config.get('datastore'))

            ds_val_data = Dataset.File.from_files((datastore, self._config.get('valdata')))
            print(' * Successfully created data input ' + self._config.get('valdata'))

        except UserErrorException:
            print(' ! ERROR:Validation Dataset not found, train dataset will be splitted')
            return None

        return ds_val_data 
    
    def _input_model(self) -> AzureAccess:

        if self._config.get('saved_model') is None:
            return None
        try:
            datastore = Datastore.get(self._workspace, self._config.get('datastore'))

            ds_model = Dataset.File.from_files((datastore, self._config.get('saved_model')))
            print(' * Successfully fechted model input ' + self._config.get('saved_model'))

        except UserErrorException:
            print(' ! ERROR:Model input not found')
            return None

        return ds_model

    def _model_ckpt(self):

        if self._config.get('ckpt') is None:
            return None

        try:
            datastore = Datastore.get(self.__ws, self._config.get('datastore'))
            model = Dataset.File.from_files((datastore, self._config.get('ckpt')))
            print(' * Successfully created model input ' + self._config.get('ckpt'))
        except:
            print('Error: Workspace not found')
            return None

        return model


    # FIXME: Should not load data input by path but name, so maybe delete.
    #def input_ds_by_path(self, datastore_name, data_path) -> AzureAccess:
    #    try:
    #        datastore = Datastore.get(self.__ws, datastore_name)
    #        self.__ds_data = Dataset.File.from_files((datastore, data_path))
    #        print('Successfully created data input ' + data_path)
    #    except UserErrorException:
    #        print('ERROR:Dataset not found')
    #        return None
    #    return self

    def _output_ds(self) -> AzureAccess:

        try:
            datastore = Datastore.get(self._workspace, self._config.get('datastore'))
            ds = OutputFileDatasetConfig(destination=(self._config.get('datastore'),
                                                  self._output_path))
            print(' * Successfully created output path to ' + self._output_path)
        except:
            print('Error: Creating output path ' + self._output_path)
            return None

        return ds

    def _set_compute_target(self):
        """Instantiate compute target. If it is local, just return the name.
        """
        if self._is_local_run():
            print(' * Run is deployed on local machine')
            return self._target

        print(' * Run will be deployed to remote compute target ' + self._target)
        return ComputeTarget(workspace=self._workspace, name=self._target)

    def _set_experiment(self):
        """Instantiate experiment
        """
        try:
            exp = Experiment(workspace=self._workspace, name=self._config.get('experiment'))
        except:
            print('Error: Can\'t instantiate experiment ' + self._config.get('experiment'))

        return exp

    def _create_snapshot(self):
        """Creates the snapshot folder, copies the src folder and the config file used
        """

        if os.path.exists(self._snapshot_folder):
            rmtree(self._snapshot_folder, ignore_errors=True)

        copytree('./src', self._snapshot_folder)
        copyfile(self._config.get('config_file'), self._snapshot_folder + '/run_config.yml')
        print(' * Snapshot folder was created')

    def register_model(self, name, model_path):
        #ws = self._get_workspace()
        Model.register(self._workspace, model_name=name, model_path=model_path)

    def run(self, script_name):

        input_arguments = []

        if self._ds_data != None:
            input_arguments.extend(['--indir', self._ds_data.as_named_input('features').as_mount()])
        if self._ds_val_data != None:
            input_arguments.extend(['--valdata', self._ds_val_data.as_named_input('validation').as_mount()])
        if self._ds_output != None:
            input_arguments.extend(['--outdir', self._ds_output.as_mount()])
        if self._ds_model_ckpt != None:
            input_arguments.extend(['--ckpt', self._ds_model_ckpt.as_named_input('model_weigts').as_mount()])
        if self._ds_model != None:
            input_arguments.extend(['--saved_model', self._ds_model.as_named_input('saved_model').as_mount()])
            
        runconf = None if self._is_local_run() else runconfig.DockerConfiguration(use_docker=True)

        config = ScriptRunConfig(
            source_directory=self._snapshot_folder,
            script=script_name,
            arguments=input_arguments,
            compute_target=self._compute_target,
            environment=self._env,
            docker_runtime_config=runconf
        )

        run = self._experiment.submit(config)
        aml_url = run.get_portal_url()

        print("Submitted to computer instance. Click link below")
        print(aml_url)