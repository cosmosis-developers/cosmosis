import abc
import numpy as np
import fcntl

output_registry = {}

class OutputMetaclass(abc.ABCMeta):
    def __init__(cls, name, b, d):
        abc.ABCMeta.__init__(cls, name, b, d)
        if name == "OutputBase":
            return
        if not name.endswith("Output"):
            raise ValueError("Output classes must be named [Name]Output")
        config_name = name.rstrip("Output").lower()

        output_registry[config_name] = cls
        if hasattr(cls, "_aliases"):
            for alias in cls._aliases:
                #Do not over-ride superclass aliases
                if alias not in output_registry:
                    output_registry[alias] = cls

class CommentFileWrapper:
    """
    This little wrapper object is to turn an OutputBase object
    into a write-only file-like object where .write commands
    are turned into comments.

    This seemed cleaner than just adding a .write function to 
    OutputBase itself since it would look like that would write directly
    to file, not as comments.
    """
    def __init__(self, obj):
        self.obj=obj
    def write(self, text):
        self.obj.comment(text)

class OutputBase(metaclass=OutputMetaclass):
    def __init__(self):
        super(OutputBase,self).__init__()
        self._columns = []
        self.closed=False
        self.begun_sampling = False
        self.resumed = False

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, columns):
        self._columns = columns

    @columns.deleter
    def columns(self, columns):
        self._columns = []

    def add_column(self, name, dtype, comment=""):
        self.columns.append((name,dtype,comment))

    def del_column(self, column):
        if isinstance(column, int):
            self.columns.pop(column)
        elif isinstance(column, tuple):
            self.columns.remove(tuple)
        elif isinstance(column, str):
            self.columns.pop(self.column_index_for_name(column))
        else:
            raise TypeError("Unknown type of column to delete")

    def column_index_for_name(self, name):
        for i,(col_name,dtype,comment) in enumerate(self._columns):
            if name==col_name:
                return i

    @property
    def column_names(self):
        return [c[0] for c in self._columns]

    def comment(self, comment):
        """
        Save a comment.  Ordering will be preserved
        if you save multiple ones.
        """
        self._write_comment(comment.strip('\n'))

    def parameters(self, *param_groups):
        """ 
        Tell the outputter to save a vector of parameters.

        You can pass in either arrays/lists, which will
        be concatenated together, or mix in some scalars
        which will be included too.  Any number of arguments
        can be used.

        A typical pattern is for a sampler to send in three
        groups: sampled variable, extra outputs, and sampler 
        outputs (e.g. likelihoods or weights).

        The number of parameters and their types
        must match what the columns are.
        """
        #Check that the length is correct
        if self.closed:
            raise RuntimeError("Tried to write parameters to closed output")

        #This is very generic
        params = []
        for p in param_groups:
            if np.isscalar(p):
                params.append(p)
            else:
                params += list(p[:])
        if not len(params)==len(self._columns):
            raise ValueError("Sampler error - tried to save wrong number of parameters, or failed to set column names")

        #If this is our first sample then 
        if not self.begun_sampling:
            self._begun_sampling(params)
            self.begun_sampling=True

        #Blind output by applying offsets
        if self._blinding_offsets is not None:
            if len(self._blinding_offsets)!=len(params):
                raise ValueError("Length of blinding offsets %d does not match number of output parameters %d"%(len(self._blinding_offsets), len(params)))
            params+= self._blinding_offsets

        #Pass to the subclasses to write output
        self._write_parameters(params)

    def reset_to_chain_start(self):
        """
        Seek the start of the chain so previous output can be overwritten.

        Subclasses can override this to actually do something.
        """
        pass

    def flush(self):
        """
        For supported output classes, flush all pending output
        """
        self._flush()

    def metadata(self, key, value, comment=""):
        """
        Save an item of metadata, with an optional comment.
        For many some output types this will only work before
        sampling has begun.
        """
        if self.closed:
            raise RuntimeError("Tried to write metadata info to closed output")
        self._write_metadata(key, value, comment)

    def final(self, key, value, comment=""):
        """
        Save an item of final metadata (like a convergence test).
        This is designed to be run after sampling has finished;
        doing otherwise might cause problems.
        """
        if self.closed:
            raise RuntimeError("Tried to write final info to closed output")
        self._write_final(key, value, comment)

    def name_for_sampler_resume_info(self):
        """
        Return a file name or directory name for an auxiliary file
        where the sampler can save information it needs to resume.

        This base class errors
        """
        raise NotImplementedError("You need to use a supported output format to use the resume feature with this sampler")


    def comment_file_wrapper(self):
        return CommentFileWrapper(self)

    @staticmethod
    def lock_file(f):
        fcntl.lockf(f, fcntl.LOCK_EX|fcntl.LOCK_NB)

    @staticmethod
    def unlock_file(f):
        fcntl.lockf(f, fcntl.LOCK_UN|fcntl.LOCK_NB)

    def close(self):
        self._close()
        self.closed=True

    def blinding_header(self):
        if self.resumed:
            return
        self.comment("")
        self.comment("Blank lines prevent accidental unblinding")
        for i in range(250):
            self.comment("")


    #These are the methods that subclasses should
    #implement.  _begun_sampling and _close are optional.
    #The others are mandatory

    def _begun_sampling(self, params):
        pass

    def _close(self):
        pass

    def _flush(self):
        pass

    @abc.abstractmethod
    def _write_parameters(self, params):
        pass

    @abc.abstractmethod
    def _write_metadata(self, key, value, comment):
        pass

    @abc.abstractmethod
    def _write_comment(self, comment):
        pass

    @abc.abstractmethod
    def _write_final(self, key, value, comment):
        pass

    @classmethod
    def from_options(cls, options, resume=False):
        """ This method should create an output object from the section of ini file it is given"""
        raise NotImplementedError("The format mode you tried to use is incomplete - sorry")

    @classmethod
    def load_from_options(cls, options):
        """ This method should load back data written using the format.
            It returns a tuple of (column_names, data, metadata, final_metadata)
            with column_names a list of strings
                 data: a list of numpy arrays (ordered by parameter, then sample)
                 metadata: a list of dictionaries
                 final_metadata: a list of dictionaries
            The latter three lists will have one element for any data chain
            identified within the data.
        """
        raise NotImplementedError("The format mode you tried to use is incomplete - sorry")
