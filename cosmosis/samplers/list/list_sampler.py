import itertools
import numpy as np
from cosmosis.output.text_output import TextColumnOutput
from .. import ParallelSampler
from ...runtime import logs

def task(p):
    i,p = p
    results = list_sampler.pipeline.run_results(p, all_params=True)
    #If requested, save the data to file
    if list_sampler.save_name and results.block is not None:
        results.block.save_to_file(list_sampler.save_name+"_%d"%i, clobber=True)
    return results.post, (results.prior, results.extra)




class ListSampler(ParallelSampler):
    parallel_output = False
    sampler_outputs = [("prior", float), ("post", float)]

    def config(self):
        global list_sampler
        list_sampler = self

        self.converged = False
        self.filename = self.read_ini("filename", str)
        self.save_name = self.read_ini("save", str, "")
        # The burn parameter can be an int or a float
        # so we read it as a string and then convert it
        burn = self.read_ini("burn", str, "0")
        try:
            self.burn = int(burn)
        except ValueError:
            self.burn = float(burn)
        self.thin = self.read_ini("thin", int, 1)
        limits = self.read_ini("limits", bool, False)
        self.chunk_size = self.read_ini("chunk_size", int, 100)

        #overwrite the parameter limits
        if not limits:
            if self.output is not None:
                self.output.columns = []
            for p in self.pipeline.parameters:
                p.limits = (-np.inf, np.inf)
                if self.output is not None:
                    self.output.add_column(str(p), float)
            if self.output is not None:
                for section,name in self.pipeline.extra_saves:
                    if ('#' in name):
                        n,l = name.split('#')
                        for i in range(int(l)):
                            self.output.add_column('{}--{}_{}'.format(section,n,i), float)
                    else:
                        self.output.add_column('%s--%s'%(section,name), float)
                for p,ptype in self.sampler_outputs:
                    self.output.add_column(p, ptype)


    def execute(self):

        #Load in the filename that was requested
        file_options = {"filename":self.filename}
        column_names, samples, _, _, _ = TextColumnOutput.load_from_options(file_options)
        samples = samples[0]
        if (self.burn != 0) or (self.thin != 1):
            if isinstance(self.burn, float):
                burn = int(self.burn*len(samples))
            else:
                burn = self.burn
            samples = samples[burn::self.thin]
        # find where in the parameter vector of the pipeline
        # each of the table parameters can be found
        replaced_params = []
        for i,column_name in enumerate(column_names):
            # ignore additional columns like e.g. "like", "weight"
            try:
                section,name = column_name.split('--')
            except ValueError:
                logs.important("Not including column %s as not a cosmosis name" % column_name)
                continue
            section = section.lower()
            name = name.lower()
            # find the parameter in the pipeline parameter vector
            # may not be in there - warn about this
            try:
                j = self.pipeline.parameters.index((section,name))
                replaced_params.append((i,j))
            except ValueError:
                logs.important("Not including column %s as not in values file" % column_name)

        #Create a collection of sample vectors at the start position.
        #This has to be a list, not an array, as it can contain integer parameters,
        #unlike most samplers
        v0 = self.pipeline.start_vector(all_params=True, as_array=False)
        sample_vectors = [v0[:] for i in range(len(samples))]
        #Fill in the varied parameters. We are not using the
        #standard parameter vector in the pipeline with its 
        #split according to the ini file
        for s, v in zip(samples, sample_vectors):
            for i,j in replaced_params:
                v[j] = s[i]

        #Turn this into a list of jobs to be run 
        #by the function above
        sample_index = list(range(len(sample_vectors)))
        jobs = list(zip(sample_index, sample_vectors))

        nsample = len(sample_index)
        nchunk = nsample//self.chunk_size
        if nsample%self.chunk_size!=0:
            nchunk += 1
        
        # Run on a chunk of parameters
        for i in range(nchunk):
            start = i*self.chunk_size
            end = (i+1)*self.chunk_size
            if end>nsample:
                end = nsample
            chunk = jobs[start:end]

            if self.pool:
                results = self.pool.map(task, chunk)
            else:
                results = list(map(task, chunk))


            # Save the results of the sampling
            # We now need to abuse the output code a little.
            for sample, result  in zip(sample_vectors[start:end], results):
                # Optionally save all the results calculated by each
                # pipeline run to files
                (prob, (prior,extra)) = result
                # always save the usual text output
                self.output.parameters(sample, extra, prior, prob)
                self.distribution_hints.set_peak(sample, prob)

        # We only ever run this once, though that could 
        # change if we decide to split up the runs
        self.converged = True

    def is_converged(self):
        return self.converged
