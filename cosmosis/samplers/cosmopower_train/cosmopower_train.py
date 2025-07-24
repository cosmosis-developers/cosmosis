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
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

# setting the seed for reproducibility
np.random.seed(1)

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

        i = os.path.basename(filename).split('_')[-1].split('.')[0]
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
                        try:
                            data[member.name.split('/')[-2]][key] = float(value)
                        except:
                            data[member.name.split('/')[-2]][key] = str(value)
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
        return i, data

    def pack_to_dictionaries(self, params, files):
        # Consider combining the two functions together!
        index_list = []
        params_dict = defaultdict(list)
        out_dict = defaultdict(lambda: defaultdict(list))
        runs = set()
       
        for tar_file in files:
            i, data = self.load_from_file(tar_file)
            # Add a cleaning step here if any are nan!
            index_list.append(i)
            for param in params:
                p = param.name
                if p in data.get('cosmological_parameters'):
                    params_dict[p].append(data['cosmological_parameters'][p])
                if p in data.get('halo_model_parameters'):
                    params_dict[p].append(data['halo_model_parameters'][p])

            for key in ['matter_power_lin', 'matter_power_nl', 'cmb_cl']:
                if data.get(key):
                    runs.add(key)
                    if key in ['matter_power_lin', 'matter_power_nl']:
                        out_dict[key]['k_h'].append(list(data[key]['k_h']))
                        out_dict[key]['p_k'].append(list(data[key]['p_k']))
                    elif key in ['cmb_cl']:
                        for cl in ['ell', 'tt', 'ee', 'bb', 'te']:
                            out_dict[key][cl].append(list(data[key][cl]))
                        for cl in ['pp', 'pt', 'pe']:
                            if cl in data[key]:
                                out_dict[key][cl].append(list(data[key][cl]))

        return index_list, list(runs), params_dict, dict(out_dict)


    def config(self):
        import tensorflow as tf     
        self.converged = False
        self.fatal_errors = self.read_ini("fatal_errors", bool, False)
        self.save_dir = self.read_ini("save_dir", str, "")
        self.print_log = self.read_ini("print_log", bool, False)

        self.device = 'gpu:0' if tf.config.list_physical_devices('GPU') else 'cpu'
        self.samples_name = self.read_ini("samples_file_name", str, "")
        self.tests_name = self.read_ini("tests_file_name", str, "")


    def execute(self):
        import matplotlib.pyplot as plt
        import tensorflow as tf    
        from cosmopower import cosmopower_NN
        # setting the seed for reproducibility
        tf.random.set_seed(2)

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
        tar_files_train = glob.glob(f"{self.samples_name}_*.tgz")
        tar_files_test  = glob.glob(f"{self.tests_name}_*.tgz")
        tar_files_train = [f for f in tar_files_train if f not in tar_files_test]

        index_list_train, training_runs, params_dict_train, runs_dict = self.pack_to_dictionaries(params, tar_files_train)
        index_list_test, test_runs, params_dict_test, test_dict = self.pack_to_dictionaries(params, tar_files_test)
            
        # outputs are either Plin, Pnl, CMB spectra
        for i, run in enumerate(training_runs):
            print(f'CosmoPower run {i+1}: training on {run}!')
            if run == 'cmb_cl':
                print('Skipping CMB training, not yet implemented!')
                continue
                
            mean_params = [np.mean(params_dict_train[key]) for key in list(params_dict_train.keys())]
            mean_run = self.pipeline.run_results(mean_params)
            reference_prediction = np.squeeze(mean_run.block[run, 'p_k'])
            # TO-DO: save reference prediction for later use!
            
            # Initiate and train the neural network
            cp_nn = cosmopower_NN(parameters=list(params_dict_train.keys()),
                                  modes=runs_dict[run]['k_h'][0],
                                  n_hidden=[1048, 1048, 1048, 1048],
                                  verbose=True)
            
            training_features = np.log10(runs_dict[run]['p_k']) - np.log10(reference_prediction)
            with tf.device(self.device):
                # The training choices are for now hardcoded. These showed to perform well for any CAMB outputs.
                cp_nn.train(training_parameters=params_dict_train,
                            training_features=training_features,
                            filename_saved_model=f'{self.save_dir}_{run}',
                            validation_split=0.1,
                            learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                            batch_sizes=[1000, 1000, 1000, 1000, 1000, 1000],
                            gradient_accumulation_steps=[1, 1, 1, 1, 1, 1],
                            patience_values=[100, 100, 100, 100, 100, 100],
                            max_epochs=[1000, 1000, 1000, 1000, 1000, 1000])
        
            # Enter testing phase
            cp_nn_test = cosmopower_NN(restore=True, restore_filename=f'{self.save_dir}_{run}')
            emulated_test_spectra = 10.0**(cp_nn_test.predictions_np(params_dict_test) + np.log10(reference_prediction))

            diff = np.abs((emulated_test_spectra - test_dict[run]['p_k']) / test_dict[run]['p_k'])
            # Define percentiles to calculate
            percentile_values = [68, 95, 99, 99.9]
            percentiles = np.array([np.percentile(diff, p, axis=0) for p in percentile_values])

            # Define colors and labels for plotting
            plot_settings = [
                {'color': 'salmon', 'label': '99%', 'alpha': 0.8},
                {'color': 'red', 'label': '95%', 'alpha': 0.7},
                {'color': 'darkred', 'label': '68%', 'alpha': 1}
            ]

            plt.figure(figsize=(12, 9))
            for i, settings in enumerate(plot_settings[::-1]):
                plt.fill_between(runs_dict[run]['k_h'][0], 0, percentiles[i], **settings)

            plt.xscale('log')
            plt.legend(frameon=False, fontsize=30, loc='upper left')
            plt.ylabel(r'$\frac{\vert P(k,z)^{\mathrm{emulated}}_\mathrm{lin} -P(k,z)^{\mathrm{true}}_\mathrm{lin} \vert} {P(k,z)^{\mathrm{true}}_\mathrm{lin} }$', fontsize=30)
            plt.xlabel(r'$k [h Mpc^{-1}]$', fontsize=30)

            # Save the plot
            plt.savefig(f'{self.save_dir}_{run}_test.jpg', dpi=200, bbox_inches='tight')
            plt.close()

        self.converged = True


    def is_converged(self):
        return self.converged
