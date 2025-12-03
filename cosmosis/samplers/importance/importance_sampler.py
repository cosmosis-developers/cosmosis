import itertools
import collections
import numpy as np
from ...runtime import logs
from ...output import TextColumnOutput, FitsOutput

from .. import ParallelSampler


INI_SECTION = "importance"


def task(p):
    r = importance_pipeline.run_results(p)
    return r.post, (r.prior, r.extra)


class ImportanceSampler(ParallelSampler):
    #there are a bunch of these really, but we 
    #need to add them manually so that we can
    #keep the likelihood column at the end
    #so the postprocessors don't need to be re-written
    sampler_outputs = []
    parallel_output = False
    supports_resume = True

    def config(self):
        global importance_pipeline
        importance_pipeline = self.pipeline
        self.input_filename = self.read_ini("input", str)
        self.nstep = self.read_ini("nstep", int, 128)
        self.add_to_likelihood = self.read_ini("add_to_likelihood", bool, False)

        self.converged = False
        if self.is_master():
            self.load_samples(self.input_filename)

    def load_samples(self, filename):
        options = {"filename":filename}
        if filename.endswith(".txt"):
            output_cls = TextColumnOutput
        elif filename.endswith(".fits"):
            output_cls = FitsOutput
        else:
            raise ValueError(f"Can only postprocess files in cosmosis text format (.txt) or fits format (.fits), not {filename}")
            
        col_names, cols, metadata, comments, final_metadata = output_cls.load_from_options(options)
        # pull out the "post" column first
        col_names = [name.lower() for name in col_names]
        posterior_index = col_names.index('post')
        if posterior_index<0:
            raise ValueError("I could not find a 'post' column in the chain %s"%filename)
        self.original_posteriors = cols[0].T[posterior_index]

        #We split the parameters into three groups:
        #   - ones that we have listed as varying
        #   - one that are fixed - WHAT SHOULD WE DO ABOUT THESE???
        #   - extras ones - these should be saved and put in the output

        self.original_extras = collections.OrderedDict()
        self.samples = []
        self.varied_params = []
        self.number_samples = len(cols[0])
        logs.overview("Have %d samples from old chain." % self.number_samples)
        for code,col in zip(col_names, cols[0].T):
            #we have already handled the likelihood
            if code=='post':continue
            #parse the header names in to (section,name)
            bits = code.split("--")
            if len(bits)==2:
                section, name = bits
                #No parameter should be varied in the old chain
                #but listed as fixed here
                if (section,name) in self.pipeline.fixed_params:
                    logs.error("WARNING: %s varied in old chain now fixed.  I will fix it <--------- Read this warning." % name)
                elif (section,name) in self.pipeline.varied_params:
                    #Record the values of this parameter for later importance
                    #sampling
                    logs.overview(f"Found column in both pipelines {code}:")
                    self.samples.append(col)
                    self.varied_params.append(bits)
                else:
                    logs.overview(f"Found column just in old pipeline: {code}")
                    #This parameter was varied in the old code but is not
                    #here.  So we just save it for output
                    self.original_extras[code] = col
                    self.output.add_column(code, float)
            # anything here must be a sampler-specific 
            else:
                logs.overview(f"Found non-parameter column: {code}")
                self.original_extras[code] = col
                if code=="weight":
                    code="old_weight"
                    logs.overview("Renaming weight -> old_weight")
                elif code == "log_weight":
                    code = "old_log_weight"
                    print("Renaming log_weight -> old_log_weight")
                self.output.add_column(code, float)

        #Now finally add our actual two sampler outputs, old_like and like
        #P is the original likelihood in the old chain we have samples from
        #P' is the new likelihood.
        self.output.add_column("old_post", float) #This is the old likelihood, log(P)
        self.output.add_column("log_weight", float) #This is the log-weight, the ratio of the likelihoods
        self.output.add_column("prior", float) #This is the new prior
        self.output.add_column("post", float) #This is the new likelihood, log(P')


        #Now we need to reorder the list to match the order our pipeline is expecting
        #If a parameter is not listed we use the starting value
        reordered_cols = []
        for p in self.pipeline.varied_params:
            try:
                i = self.varied_params.index(p)
            except ValueError:
                i=-1
            if i>=0:
                col = self.samples[i]
            else:
                col = np.repeat(p.start,self.number_samples)
            reordered_cols.append(col)
        self.samples = np.transpose(reordered_cols)

        self.current_index = 0

    def resume(self):
        if self.output.resumed:
            data = np.genfromtxt(self.output._filename, invalid_raise=False)
            self.current_index = len(data)
            if self.current_index >= self.number_samples:
                logs.error(f"You told me to resume the chain, but it has already completed (with {self.current_index} samples).")
            else:
                logs.overview(f"Continuing importance sampling from existing chain - have {self.current_index} samples already")


    def execute(self):

        #Pick out a chunk of samples to run on
        start = self.current_index
        end = start+self.nstep
        samples_chunk = self.samples[start:end]

        logs.overview(f"Importance sampling {start} - {end} of {self.number_samples} from {self.input_filename}")

        #Run the pipeline on each of the samples
        if self.pool:
            results = self.pool.map(task, samples_chunk)
        else:
            results = list(map(task, samples_chunk))

        #Collect together and output the results
        for i,(sample, (new_post, extra)) in enumerate(zip(samples_chunk, results)):
            #We already (may) have some extra values from the pipeline
            #as derived parameters.  Add to those any parameters used in the
            #old pipeline but not the new one
            new_prior, extra = extra
            extra = list(extra)
            for col in list(self.original_extras.values()):
                extra.append(col[start+i])
            #and then the old and new likelihoods
            old_post = self.original_posteriors[start+i]
            if self.add_to_likelihood:
                new_post += old_post
            weight = new_post - old_post
            extra = list(extra) + [old_post, weight, new_prior, new_post]
            #and save results
            self.output.parameters(sample, extra)
            self.distribution_hints.set_peak(sample, new_post)
        #Update the current index
        self.current_index+=self.nstep

    def is_converged(self):
        return self.current_index>=self.number_samples
