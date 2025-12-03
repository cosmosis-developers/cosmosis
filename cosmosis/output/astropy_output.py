from .output_base import OutputBase

class AstropyOutput(OutputBase):
    _aliases = ["astropy"]
    def __init__(self):
        super(AstropyOutput,self).__init__()
        from astropy.table import Table
        self.table = Table()

    def _begun_sampling(self, params):
        from astropy.table import Column
        for name, dtype, comment in self.columns:
            self.table.add_column(Column(name=name, dtype=dtype))

    def _write_parameters(self, params):
        self.table.add_row(params)

    def __getitem__(self, key_or_index):
        return self.table[key_or_index]

    def _write_metadata(self, key, value, comment):
            self.table.meta[key] = value
            if comment:
                self.table.meta[key + "_comment"] = comment

    def _write_final(self, key, value, comment):
            self.table.meta["final:" + key] = value
            if comment:
                self.table.meta["final:" + key + "_comment"] = comment

    def _write_comment(self, comment):
        self.table.meta["comments"] = self.table.meta.get("comments", []) + [comment]

    def reset_to_chain_start(self):
         self.table.remove_rows(slice(None))

