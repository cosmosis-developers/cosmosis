from .output_base import OutputBase
from . import utils
import numpy as np
import os
from glob import glob
from collections import OrderedDict
try:
    import fitsio
except ImportError:
    fitsio = None

comment_indicator = "_cosmosis_comment_indicator_"
final_metadata_indicator = "FINALMETA"
unreserve_indicator = "UNRES"
reserved_keys = [
    "XTENSION",
    "BITPIX",
    "NAXIS",
    "NAXIS1",
    "NAXIS2",
    "PCOUNT",
    "GCOUNT",
    "TFIELDS",
    "TTYPE1",
    "COMMENT",
]

def check_fitsio():
    if fitsio is None:
        raise RuntimeError("You need to have the fitsio library installed to output FITS files. Try running: pip install  --install-option=\"--use-system-fitsio\" git+git://github.com/joezuntz/fitsio")

class FitsOutput(OutputBase):
    FILE_EXTENSION = ".fits"
    _aliases = ["fits"]

    def __init__(self, filename, rank=0, nchain=1, clobber=True, blinding_offset_file=None):
        super(FitsOutput, self).__init__()

        #If filename already ends in .txt then remove it for a moment
        if filename.endswith(self.FILE_EXTENSION):
            filename = filename[:-len(self.FILE_EXTENSION)]

        if nchain > 1:
            filename = filename + "_{}".format(rank+1)

        self._filename = filename + self.FILE_EXTENSION

        self.filename_base = filename

        check_fitsio()

        self._fits = fitsio.FITS(self._filename, "rw", clobber=clobber)
        self._hdu = None

        #also used to store comments:
        self._metadata = OrderedDict()
        self._final_metadata = OrderedDict()

        if blinding_offset_file is not None:
            self._blinding_offsets = np.load(blinding_offset_file)
        else:
            self._blinding_offsets = None

    def _close(self):
        self._flush_metadata(self._final_metadata)
        self._final_metadata={}
        self._fits.close()

    def _flush_metadata(self, metadata):

        for (key,(value,comment)) in list(metadata.items()):
            if key.startswith(comment_indicator):
                self._hdu.write_comment(value)
            elif comment:
                self._hdu.write_key(key, value, comment)
            else:
                self._hdu.write_key(key, value)


    def _begun_sampling(self, params):
        #write the name line
        self._fits.create_table_hdu(data=params, names=[c[0] for c in self.columns])
        self._hdu = self._fits[-1]
        self._dtype = self._hdu.get_rec_dtype()[0]
        self._flush_metadata(self._metadata)
        self._metadata={}

    @staticmethod
    def is_reserved_fits_keyword(key):
        for k in reserved_keys:
            if key.upper().startswith(k):
                return True
        return False

    def _write_metadata(self, key, value, comment=''):
        #We save the metadata until we get the first 
        #parameters since up till then the columns can
        #be changed
        if self.is_reserved_fits_keyword(key):
            key=unreserve_indicator + key
        self._metadata[key]= (value, comment)

    def _write_comment(self, comment):
        #save comments along with the metadata - nice as 
        #preserves order
        self._metadata[comment_indicator +
                       "_%d" % (len(self._metadata))] = (comment,None)

    def _write_parameters(self, params):
        row = np.core.records.fromarrays(params, dtype=self._dtype)
        row=np.atleast_1d(row)
        self._hdu.append(row)

    def _write_final(self, key, value, comment=''):
        #I suppose we can put this at the end - why not?
        if self.is_reserved_fits_keyword(key):
            key=unreserve_indicator + key
        self._final_metadata[key]= (value, final_metadata_indicator+comment)

    def name_for_sampler_resume_info(self):
        return self.filename_base + '.sampler_status'

    @classmethod
    def from_options(cls, options, resume=False):
        #look something up required parameters in the ini file.
        #how this looks will depend on the ini 
        if resume:
            raise ValueError("Cannot resume from FITS output")
            
        filename = options['filename']
        delimiter = options.get('delimiter', '\t')
        rank = options.get('rank', 0)
        nchain = options.get('parallel', 1)
        blinding_offset_file = options.get('blinding_offsets', None)
        clobber = utils.boolean_string(options.get('clobber', True))
        return cls(filename, rank, nchain, clobber=clobber, blinding_offset_file=blinding_offset_file)

    @classmethod
    def load_from_options(cls, options):
        check_fitsio()
        filename = options['filename']

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

            f = fitsio.FITS(datafile, "r")
            hdu = f[1]
            chain = f[1].read()
            #convert to unstructured format
            chain = chain.view((chain.dtype[0], len(chain.dtype.names)))

            column_names = hdu.get_colnames()

            hdr = hdu.read_header()
            chain_comments = [r['comment'] for r in hdr.records() if r['name'].lower()=="comment"]
            for r in hdr.records():
                key = r['name']
                if key=='COMMENT':
                    continue
                if key.startswith(unreserve_indicator):
                    key = key[len(unreserve_indicator):]
                value = r['value']
                key=key.lower()
                if r['comment'].startswith(final_metadata_indicator):
                    chain_final_metadata[key] = value
                else:
                    chain_metadata[key] = value

            data.append(np.array(chain))
            metadata.append(chain_metadata)
            final_metadata.append(chain_final_metadata)
            comments.append(chain_comments)

        if column_names is None:
            raise ValueError("Could not find column names header in file starting %s"%filename)

        return column_names, data, metadata, comments, final_metadata
