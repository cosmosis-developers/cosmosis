from .. import Sampler
import numpy as np
from ...runtime import pipeline
import sys
import os
import io
import glob
import tarfile
from collections import defaultdict
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from cosmopower import cosmopower_NN
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# setting the seed for reproducibility
np.random.seed(1)
tf.random.set_seed(2)

class CosmoPowerTrainingSampler(Sampler):
    parallel_output = False
    needs_output = False

    def load_from_file(self, filename):
        """Loads and decompresses a .tgz file, extracting its contents into memory.

        Args:
            filename (str): The name of the .tgz file to be read.

        Returns:
            dict: A dictionary containing the extracted data.
        """
        if not filename.endswith('.tgz'):
            raise ValueError("The file must have a .tgz extension.")

        if not os.path.exists(filename):
            raise FileNotFoundError(f"The file {filename} does not exist.")

        data = defaultdict(dict)

        with tarfile.open(filename, "r:gz") as tar:
            for member in tar.getmembers():
                file = tar.extractfile(member)
                content = file.read().decode('utf-8')
                lines = content.splitlines()

                # Determine if it's a scalar or vector file
                if member.name.endswith('values.txt'):
                    # Handle scalar outputs
                    for line in lines:
                        if line.startswith('#'):
                            continue  # Skip comments or metadata for now
                        if '=' in line:
                            key, value = line.split(' = ')
                        data[member.name.split('/')[-2]][key] = value
                else:
                    # Handle vector outputs
                    header = lines[0]
                    header_lines = [line for line in lines if line.startswith(header) or '=' in line]
                    header_info = {}
                    for line in header_lines[1:]:
                        if '=' in line:
                            key, value = line.split(' = ')
                            header_info[key] = value
                    # Extract data
                    data_values = [line for line in lines if not line.startswith(header) and '=' not in line]
                    if data_values:
                        vector_data = np.loadtxt(io.StringIO('\n'.join(data_values)))
                        data[member.name.split('/')[-2]][member.name.split('/')[-1].split('.')[0]] = vector_data
        return data


    def config(self):
        self.converged = False
        self.fatal_errors = self.read_ini("fatal_errors", bool, False)
        self.save_dir = self.read_ini("save_dir", str, "")
        self.print_log = self.read_ini("print_log", bool, False)

        self.device = 'gpu:0' if tf.config.list_physical_devices('GPU') else 'cpu'
        self.samples_name = self.read_ini("samples_file_name", str, "")
        self.tests_name = self.read_ini("tests_file_name", str, "")


    def execute(self):

        # If our pipeline allows it we arrange it so that the
        # fast parameters change fastest in the sequence.
        # This is still not optimal for the multiprocessing case
        if self.pipeline.do_fast_slow:
            params = self.pipeline.slow_params + self.pipeline.fast_params
        else:
            params = self.pipeline.varied_params

        # We load the generated power spectra but interpolate them over specific k-range as defined here.
        # This makes sure our training has more features where power spectra change faster ...
        # How to deal with redshifts?? Ideally as done by Pierre, we train at specific z's that are also in latin hypercube
        # CAMB by default returns power spectra at an vector. 

        # Load your data
        # Add check if any spectra are nan and throw out the indices!
        # need to loop through indices as saved by the LHS
        tar_files = glob.glob(f"{self.samples_name}_*.tgz")
        nsamples = len(tar_files)

        print(self.samples_name)
        print(self.tests_name)
        print(tar_files)
        print(nsamples)




        params_cube = {}
        for tar_file in tar_files:
            i = os.path.basename(tar_file).split('_')[-1].split('.')[0]
            data = self.load_from_file(tar_file)
            for key, item in data.items():
                for key2, item2 in item.items():
                    print(key, key2, item2)
            quit()
            #print(data.keys())
            #print(data['./runs/cosmopower_inputs_0/halo_model_parameters/values.txt'])
            #params_cube = data[...]
            #k_modes = data[...]
            #camb_outputs = data[...]
            #reference_output = data[...]

        tar_files_test = glob.glob(f"{self.tests_name}_*.tgz")
        nsamples_test = len(tar_files_test)
        for tar_file in tar_files_test:
            i = os.path.basename(tar_file).split('_')[-1].split('.')[0]
            data_test = self.load_from_file(tar_file)
            #print(data_test)
            #camb_outputs_test = data_test[...]
            #reference_output_test = data_test[...]

        raise SystemExit()

        params_dict = {}
        for param in params:
            params_dict[param] = params_cube[param]

        # outputs are either Plin, Pnl, CMB spectra
        for outputs in camb_outputs:

            # Instantiate and train the neural network
            cp_nn = cosmopower_NN(parameters=params,
                                  modes=k_modes,
                                  n_hidden=[1048, 1048, 1048, 1048],
                                  verbose=True)

            training_features = np.log10(outputs) - np.log10(reference_output)
            with tf.device(self.device):
                # The training choices are for now hardcoded. These showed to perform well for any CAMB outputs.
                cp_nn.train(training_parameters=params_dict,
                            training_features=training_features,
                            filename_saved_model=f'{self.save_dir}_{self.power_spectra_version}', #exact nameing tbd!
                            validation_split=0.1,
                            learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                            batch_sizes=[1000, 1000, 1000, 1000, 1000, 1000],
                            gradient_accumulation_steps=[1, 1, 1, 1, 1, 1],
                            patience_values=[100, 100, 100, 100, 100, 100],
                            max_epochs=[1000, 1000, 1000, 1000, 1000, 1000])

        
        try:
            # Use this to save test figures!
            if self.save_dir:
                if data is not None:
                    if self.save_dir.endswith('.tgz'):
                        data.save_to_file(self.save_dir[:-4], clobber=True)
                    else:
                        data.save_to_directory(self.save_dir, clobber=True)
                else:
                    print("(There was an error so no output to save)")
        except Exception as e:
            if self.fatal_errors:
                raise
            print("Could not save output.")

        if data is None and self.fatal_errors:
            raise RuntimeError("CosmoPower failed at some stage")

        if self.print_log and data is not None:
            data.print_log()

        self.converged = True


    def is_converged(self):
        return self.converged
