from .. import Sampler
import numpy as np
from ...runtime import pipeline
import sys
import os
import io
import glob
import pickle
from collections import defaultdict
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

# setting the seed for reproducibility
np.random.seed(1)

matter_power_section_names = [
    'dark_matter_power',
    'baryon_power',
    'photon_power',
    'massless_neutrino_power',
    'massive_neutrino_power',
    'matter_power',
    'cdm_baryon_power',
    'matter_de_power',
    'weyl_curvature_power',
    'cdm_velocity_power',
    'baryon_velocity_power',
    'baryon_cdm_relative_velocity_power',
]

class CosmoPowerSampler(Sampler):
    parallel_output = False
    needs_output = False

    def load_pickles(self, params, filenames):
        index_list = []
        params_dict = defaultdict(list)
        out_dict = defaultdict(lambda: defaultdict(list))
        runs = set()

        for filename in filenames:
            index_list.append(int(os.path.basename(filename).split('_')[-1].split('.')[0]))
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        
            # This could be simplified and generalised for all inputs, by having the same function as block.from_pickle, 
            # but leaving now as it is to extract only the relevant quantities
            for param in params:
                p = param.name
                # Extend the option where the parameters can be and which spectra we have!
                for key in ['cosmological_parameters', 'halo_model_parameters', 'redshift_as_parameter']:
                    if p in data.get(key):
                        params_dict[p].append(data[key][p])

            suffixes = ['lin', 'nl']
            all_keys = [f'{key}_{suffix}' for key in matter_power_section_names for suffix in suffixes]
            all_keys.append('cmb_cl')
            for key in all_keys:
                if data.get(key):
                    runs.add(key)
                    if key != 'cmb_cl':
                        out_dict[key]['k_h'].append(np.squeeze(data[key]['k_h']).tolist())
                        out_dict[key]['p_k'].append(np.squeeze(data[key]['p_k']).tolist())
                    elif key == 'cmb_cl':
                        for cl in ['ell', 'tt', 'ee', 'bb', 'te']:
                            out_dict[key][cl].append(np.squeeze(data[key][cl]).tolist())
                        for cl in ['pp', 'pt', 'pe']:
                            if cl in data[key]:
                                out_dict[key][cl].append(np.squeeze(data[key][cl]).tolist())

        return index_list, list(runs), params_dict, dict(out_dict)

    def config(self):
        import tensorflow as tf     
        self.converged = False
        self.fatal_errors = self.read_ini("fatal_errors", bool, False)
        self.print_log = self.read_ini("print_log", bool, False)

        self.device = 'gpu:0' if tf.config.list_physical_devices('GPU') else 'cpu'

        root_dir_name = self.ini.get("training", "save_dir")
        self.save_dir = f'{root_dir_name}/cosmopower_emulator'
        self.samples_name = f'{root_dir_name}/cosmopower_inputs'
        self.tests_name =  f'{root_dir_name}/cosmopower_inputs_test'

    def execute(self):
        import matplotlib.pyplot as plt
        import tensorflow as tf    
        from cosmopower import cosmopower_NN, cosmopower_PCAplusNN
        # setting the seed for reproducibility
        tf.random.set_seed(2)

        def run_training_pk(run, params_dict_train, runs_dict, mean_run):
            reference_prediction = np.squeeze(mean_run.block[run, 'p_k'])
            # Save reference prediction to file
            with open(f'{self.save_dir}_{run}_reference.pkl', 'wb') as f:
                pickle.dump(reference_prediction, f)
            
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
        
        def run_testing_pk(run, params_dict_test, test_dict, mean_run):
            reference_prediction = np.squeeze(mean_run.block[run, 'p_k'])
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

        def run_training_cmb():
            raise NotImplementedError

        def run_testing_cmb():
            raise NotImplementedError

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
        files_train = glob.glob(f"{self.samples_name}_*.pkl")
        files_test  = glob.glob(f"{self.tests_name}_*.pkl")
        files_train = [f for f in files_train if f not in files_test]

        index_list_train, training_runs, params_dict_train, runs_dict = self.load_pickles(params, files_train)
        index_list_test, test_runs, params_dict_test, test_dict = self.load_pickles(params, files_test)

        assert set(training_runs) == set(test_runs), "The training and testing runs do not contain the same elements!"
            
        # Save fixed parameters to file that were used to generate power spectra.
        # This is useful later when we need to check if the input parameters are the same when using the emulator.
        fixed_params = {p.name: p.limits[0] for p in self.pipeline.fixed_params}
        with open(f'{self.save_dir}_fixed_params.pkl', 'wb') as f:
            pickle.dump(fixed_params, f)

        limits = {p.name: p.limits for p in params}
        with open(f'{self.save_dir}_param_limits.pkl', 'wb') as f:
            pickle.dump(limits, f)

        mean_params = [np.mean(params_dict_train[key]) for key in list(params_dict_train.keys())]
        mean_run = self.pipeline.run_results(mean_params)

        # outputs are either Plin, Pnl, CMB spectra
        for i, run in enumerate(training_runs):
            print(f'CosmoPower run {i+1}: training on {run}!')
            if run == 'cmb_cl':
                #run_training_cmb()
                #run_testing_cmb()
                print('Skipping CMB training, not yet implemented!')
                continue
            else:    
                run_training_pk(run, params_dict_train, runs_dict, mean_run)
                run_testing_pk(run, params_dict_test, test_dict, mean_run)

        self.converged = True


    def is_converged(self):
        return self.converged
