from .text_output import TextColumnOutput
import numpy as np
import os
from glob import glob
from collections import OrderedDict
import numpy as np

PARAM_NAME = '.paramnames'


class CosmoMCOutput(TextColumnOutput):
    def __init__(self, filename, rank=0, nchain=1, delimiter='    ', lock=True, resume=False, blinding_offset_file=None):
        super(CosmoMCOutput, self).__init__(filename, rank, nchain, '', lock=lock, resume=resume)
        if filename.endswith(self.FILE_EXTENSION):
            filename = filename[:-len(self.FILE_EXTENSION)]
        if rank == 0: 
            self._paramfile = open(filename+PARAM_NAME, 'w')
        else:
            self._paramfile = None
        self._last_params = None
        self._multiplicity = 0
        if blinding_offset_file is not None:
            self._blinding_offsets = np.load(blinding_offset_file)
        else:
            self._blinding_offsets = None

    def _close(self):
        self._write_parameters_multiplicity()
        super(CosmoMCOutput, self)._close()
        if self._paramfile:
            self._paramfile.close()

    def _begun_sampling(self, params):
        if self._paramfile:
            if self.columns[-1][0].upper() != "POST":
                raise RuntimeError("CosmoMC output format assumes "
                                       "likelihood is last column.")
            for c in self.columns[:-1]:
                self._paramfile.write(c[0]+'\n')
            self._paramfile.close()
        self._metadata=OrderedDict()

    def _write_comment(self, comment):
        #Do not think cosmomc can handle comments
        pass

    def _write_parameters(self, params):
        if (self._last_params is not None) and all([p==q for (p,q) in zip(self._last_params,params)]):
            self._multiplicity += 1
        else:
            self._write_parameters_multiplicity()
            self._last_params = params[:]
            self._multiplicity = 1
    
    def _write_parameters_multiplicity(self):
        if self._last_params:
            post = self._last_params[-1]
            line = self.delimiter.join(('%16.7E'%x) for x
                                       in ([self._multiplicity, post] +
                                           self._last_params[:-1])) + '\n'
            self._file.write(line)

    @classmethod
    def load_from_options(cls, options):
        filename = options['filename']

        if filename.endswith(cls.FILE_EXTENSION):
            filename = filename[:-len(cls.FILE_EXTENSION)]

        # read column names from parameterfile
        column_names = [line.split()[0] for line in open(filename+PARAM_NAME)]
        column_names.append("LIKE")

        # first look for serial file
        if os.path.exists(filename+cls.FILE_EXTENSION):
            datafiles = [filename+cls.FILE_EXTENSION]
        else:
            datafiles = glob(filename+"_[0-9]*"+cls.FILE_EXTENSION)
            if not datafiles:
                raise RuntimeError("No datafiles found!")

        # cosmomc has no metadata support
        metadata = final_metadata = [{}]*len(datafiles)
        comments = []

        data = []
        for datafile in datafiles:
            print('LOADING CHAIN FROM FILE: ', datafile)
            chain = []
            with open(datafile) as f:
                for line in f:
                    vals = [float(word) for word in line.split()]
                    for i in range(int(vals[0])):
                        chain.append(vals[1:])
                chain = np.array(chain)
            print(datafile, chain.shape)
            data.append(chain)
        return column_names, data, metadata, comments, final_metadata

    @classmethod
    def from_options(cls, options, resume=False):
        if resume:
            raise ValueError("Cannot resume from cosmomc files yet")
        blinding_offset_file = options.get('blinding_offsets', None)
        return super(CosmoMCOutput,cls).from_options(options, resume=False, blinding_offset_file=blinding_offset_file)

