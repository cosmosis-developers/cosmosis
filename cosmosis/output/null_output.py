from .output_base import OutputBase

class NullOutput(OutputBase):
    _aliases = ["none"]

    def __init__(self, blinding_offset_file=None):
        if blinding_offset_file is not None:
            self._blinding_offsets = np.load(blinding_offset_file)
        else:
            self._blinding_offsets = None

    def _write_parameters(self, params):
        pass

    def _write_metadata(self, key, value, comment):
        pass

    def _write_comment(self, comment):
        pass

    def _write_final(self, key, value, comment):
        pass

    @classmethod
    def from_options(cls, options, resume=False):
        if resume:
            raise ValueError("Cannot resume from null output")
        blinding_offset_file = options.get('blinding_offsets', None)
        return cls(blinding_offset_file=blinding_offset_file)

    @classmethod
    def load_from_options(cls, options):
        raise ValueError("No output was saved from this run")
