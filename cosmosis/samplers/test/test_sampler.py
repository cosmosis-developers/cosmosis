from .. import Sampler
import numpy as np
from ...runtime import pipeline
import sys


class TestSampler(Sampler):
    # The name of this class makes pytest think that it is a
    # test to be run, instead of a tool for testing likelhioods.
    # This should stop that from happening.
    __test__ = False

    needs_output = False

    def config(self):
        self.converged = False
        self.fatal_errors = self.read_ini("fatal_errors", bool, False)
        self.save_dir = self.read_ini("save_dir", str, "")
        self.graph = self.read_ini("graph", str, "")
        self.analyze_fast_slow = self.read_ini("analyze_fast_slow", bool, False)
        self.print_log = self.read_ini("print_log", bool, False)

    def execute(self):
        # load initial parameter values
        p = self.start_estimate()
    
        # try to print likelihood if it exists
        data=None
        try:
            prior = self.pipeline.prior(p)
            like, extra, data = self.pipeline.likelihood(p, return_data=True)
            if self.pipeline.likelihood_names:
                print("Prior      = ", prior)
                print("Likelihood = ", like)
                print("Posterior  = ", like+prior)
        except pipeline.MissingLikelihoodError as error:
            found_likelihoods = [k[1][:-5] for k in list(error.pipeline_data.keys()) if k[0]=="likelihoods"]            
            sys.stderr.write("\n")
            sys.stderr.write("One of the likelihoods you asked for was not found.\n")
            sys.stderr.write("You asked for: %s\n"%str(error))
            sys.stderr.write("But the only ones calculated in the pipeline were:\n")
            sys.stderr.write(", ".join(found_likelihoods)+"\n")
            sys.stderr.write("\n")
            if self.fatal_errors:
                raise
        except Exception as e:
            if self.fatal_errors:
                raise
            print("(Could not get a likelihood) Error:"+str(e))
        if not self.pipeline.likelihood_names:
            print("(No likelihoods required in ini file)")
            print()

        if self.analyze_fast_slow:
            print("\n")
            self.pipeline.do_fast_slow = True
            self.pipeline.setup_fast_subspaces(all_params=True)
            print("\n")


        if self.graph:
            self.pipeline.make_graph(data, self.graph)

        try:
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
            raise RuntimeError("Pipeline failed at some stage")

        if self.print_log and data is not None:
            data.print_log()


        self.converged = True

    def is_converged(self):
        return self.converged
