from .output_base import OutputBase
from . import utils
from ..runtime.utils import mkdir
import numpy as np
import os
from glob import glob
from collections import OrderedDict

comment_indicator = "_cosmosis_comment_indicator_"


class TextColumnOutput(OutputBase):
    FILE_EXTENSION = ".txt"
    _aliases = ["text", "txt"]

    def __init__(self, filename, rank=0, nchain=1, delimiter='\t', lock=True, resume=False):
        super(TextColumnOutput, self).__init__()
        self.delimiter = delimiter

        #If filename already ends in .txt then remove it for a moment
        if filename.endswith(self.FILE_EXTENSION):
            filename = filename[:-len(self.FILE_EXTENSION)]

        if nchain > 1:
            filename = filename + "_{}".format(rank+1)

        self.filename_base = filename

        self._filename = filename + self.FILE_EXTENSION

        dirname, _ = os.path.split(self._filename)
        mkdir(dirname)

        self._any_written = False
        if resume and utils.file_exists_and_is_empty(self._filename):
            print("* You set resume=T but the file {} is empty so I will start afresh".format(self._filename))
            self._file = open(self._filename, "w")
            self.resumed = False
        elif resume and os.path.exists(self._filename):
            print("* Note: You set resume=T so I will resume from file {}".format(self._filename))
            self._file = open(self._filename, "r+")
            # Jump to the end of the file
            self._file.seek(0,2)
            self.resumed = True
            self._any_written = True

        else:
            if resume:
                print("* Note: You set resume=T but the file {} does not exist so I will start a new one".format(self._filename))
            self._file = open(self._filename, "w")
            self.resumed = False
        if lock:
            try:
                self.lock_file(self._file)
            except IOError:
                error_msg = """
Another CosmoSIS process was trying to use the same output file (%s). 
This means one of four things:
1) you were trying to use MPI but left out the --mpi flag
2) your MPI installation is not working properly
3) you have another CosmoSIS run going trying to use the same filename
4) your file system cannot cope with file locks properly.  
In the last case you can set lock=F in the [output] section to disable this feature.
""" % self._filename
                raise IOError(error_msg)

        #also used to store comments:
        self._metadata = OrderedDict()
        self._final_metadata = OrderedDict()

        # For resetting
        self._start_mark = None

    def _close(self):
        self._flush_metadata(self._final_metadata)
        self._final_metadata={}
        self._file.close()

    def _flush_metadata(self, metadata):
        for (key,(value,comment)) in list(metadata.items()):
            if key.startswith(comment_indicator):
                self._file.write("## %s\n"%value.strip())
            elif comment:
                self._file.write('#{k}={v} #{c}\n'.format(k=key,v=value,c=comment))
            else:
                self._file.write('#{k}={v}\n'.format(k=key,v=value,c=comment))


    def _begun_sampling(self, params):
        #write the name line
        if not self.resumed:
            name_line = '#'+self.delimiter.join(c[0] for c in self.columns) + '\n'
            self._file.write(name_line)
            #now write any metadata.
            #text mode does not support comments
            self._flush_metadata(self._metadata)
        self._metadata={}
        self._start_mark = self._file.tell()

    def _write_metadata(self, key, value, comment=''):
        #We save the metadata until we get the first 
        #parameters since up till then the columns can
        #be changed
        #In the text mode we cannot write more metadata
        #after sampling has begun (because it goes at the top).
        #What should we do?
        self._metadata[key]= (value, comment)

    def _write_comment(self, comment):
        #save comments along with the metadata - nice as 
        #preserves order
        # remove line breaks in comments
        comment = comment.replace("\n", " ")
        self._metadata[comment_indicator +
                       "_%d" % (len(self._metadata))] = (comment,None)

    def _write_parameters(self, params):
        self._any_written = True
        line = self.delimiter.join(str(x) for x in params) + '\n'
        self._file.write(line)

    def _write_final(self, key, value, comment=''):
        #I suppose we can put this at the end - why not?
        if self._any_written:
            self._final_metadata[key]= (value, comment)

    def _flush(self):
        self._file.flush()

    def reset_to_chain_start(self):
        # On the first iteration the start mark is not set until we call
        # output the first time.
        if self._start_mark is None:
            return
        self._file.seek(self._start_mark)
        self._file.truncate()
        self._file.flush()

    def name_for_sampler_resume_info(self):
        return self.filename_base + '.sampler_status'



    @classmethod
    def from_options(cls, options, resume=False):
        #look something up required parameters in the ini file.
        #how this looks will depend on the ini 
        filename = options['filename']
        delimiter = options.get('delimiter', '\t')
        rank = options.get('rank', 0)
        nchain = options.get('parallel', 1)
        lock = utils.boolean_string(options.get('lock', True))
        return cls(filename, rank, nchain, delimiter=delimiter, lock=lock, resume=resume)

    @classmethod
    def load_from_options(cls, options):
        filename = options['filename']
        delimiter = options.get('delimiter', None)

        cut = False
        if filename.endswith(cls.FILE_EXTENSION):
            filename = filename[:-len(cls.FILE_EXTENSION)]
            cut = True
        # first look for serial file
        if os.path.exists(filename+cls.FILE_EXTENSION):
            datafiles = [filename+cls.FILE_EXTENSION]
        elif os.path.exists(filename) and not cut:
            datafiles = [filename]
        else:
            datafiles = glob(filename+"_[0-9]*"+cls.FILE_EXTENSION)
            if not datafiles:
                raise RuntimeError("No datafiles found starting with %s!"%filename)

        #Read the metadata
        started_data = False
        metadata = []
        final_metadata = []
        data = []
        comments = []
        column_names = None

        for datafile in datafiles:
            print('LOADING CHAIN FROM FILE: ', datafile)
            chain = []
            chain_metadata = {}
            chain_final_metadata = {}
            chain_comments = []
            for i,line in enumerate(open(datafile)):
                line = line.strip()
                if not line: continue
                if line.startswith('#'):
                    #remove the first #
                    #if there is another then this is a comment,
                    #not metadata
                    line=line[1:]
                    if i == 0:
                        column_names = line.split()
                    elif line.startswith('#'):
                        comment = line[1:]
                        chain_comments.append(comment)
                    else:
                        #parse form '#key=value #comment'
                        if line.count('#') == 0:
                            key_val = line.strip()
                            comment = ''
                        else:
                            key_val, comment = line.split('#', 1)
                        key,val = key_val.split('=',1)
                        val = utils.parse_value(val)
                        if started_data:
                            chain_final_metadata[key] = val
                        else:
                            chain_metadata[key] = val
                else:
                    started_data = True
                    words = line.split(delimiter)
                    vals = [float(word) for word in words]
                    chain.append(vals)
            ncol = len(column_names)
            line_lengths = np.array([len(row) for row in chain])
            #strip off the last line if it is incompletely written as often
            #the chain is interrupted
            if line_lengths[-1]!=ncol:
                print("Skipping last line of chain as it seems to have been cut off")
                print("This could conceivably cause problems for some samplers, though")
                print("not the ones like metropolis and emcee where it is most likely to happen.")
                print("If any more lines have the wrong length then this will raise an error.")
                print()
                print("You should probably check the final lines of the other files for errors")
                print("that are harder to detect, like values being truncated.")
                print()
                chain = chain[:-1]
                line_lengths = line_lengths[:-1]
            #if any more are the wrong length then something has gone wrong:
            if np.any(line_lengths!=ncol):
                raise ValueError("Your chain file is corrupted somehow: not all the lines have {} columns".format(ncol))


            data.append(np.array(chain))
            metadata.append(chain_metadata)
            final_metadata.append(chain_final_metadata)
            comments.append(chain_comments)

        if column_names is None:
            raise ValueError("Could not find column names header in file starting %s"%filename)

        return column_names, data, metadata, comments, final_metadata
