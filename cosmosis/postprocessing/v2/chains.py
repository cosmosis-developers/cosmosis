"""
This is the core of what will at some point become the new postprocessing
system in CosmoSIS.  The plan is:
- do plotting with GetDist instead of manually
- make the plotting much simpler than the complex hierarchies we have now
- connect this to the campaign system to make it easy to specify what to do after completing a campaign

These APIs will change and should not yet be relied on.

"""
from cosmosis.runtime.utils import import_by_path
from cosmosis import Inifile, output as output_module
import numpy as np
from io import StringIO
import os


class Chain:
    chain_subclasses = {}

    def __init__(
        self,
        name: str,
        sampler: str,
        columns: list[str],
        data: list[np.ndarray],
        metadata: list[dict[str, str]],
        comments: list[list[str]],
        final_metadata: list[list[str]],
        **options
    ):
        self.name = name
        self.sampler = sampler
        self.colnames = columns
        self.data = data
        self.metadata = metadata
        self.comments = comments
        self.final_metadata = final_metadata
        self.options = options
        self._mcsamples = None

    @classmethod
    def load(cls, inputs, name=None, **options):
        if isinstance(inputs, Inifile):
            name_, sampler, colnames, data, metadata, comments, final_metadata = (
                cls.load_ini(inputs)
            )
        elif isinstance(inputs, str):
            name_, sampler, colnames, data, metadata, comments, final_metadata = (
                cls.load_text_file(inputs)
            )
        elif isinstance(inputs, output_module.InMemoryOutput):
            name_, sampler, colnames, data, metadata, comments, final_metadata = (
                cls.load_in_memory_storage(inputs)
            )
        elif isinstance(inputs, output_module.AstropyOutput):
            name_, sampler, colnames, data, metadata, comments, final_metadata = (
                cls.load_astropy(inputs)
            )
        elif isinstance(inputs, output_module.OutputBase):
            name_, sampler, colnames, data, metadata, comments, final_metadata = (
                cls.load_output_object(inputs)
            )
        else:
            raise ValueError("Unknown input chain type " + str(type(inputs)))

        subclass = cls.chain_subclasses.get(sampler, cls)

        if name is None:
            name = name_

        # Assume metropolis if no more info is given
        if subclass is None:
            subclass = MetropolisChain

        return subclass(
            name, sampler, colnames, data, metadata, comments, final_metadata, **options
        )

    @classmethod
    def load_ini(cls, inputs):
        output_options = dict(inputs.items("output"))
        filename = output_options["filename"]
        name = filename
        sampler = inputs.get("runtime", "sampler")
        colnames, data, metadata, comments, final_metadata = (
            output_module.input_from_options(output_options)
        )
        return name, sampler, colnames, data, metadata, comments, final_metadata

    @classmethod
    def load_in_memory_storage(cls, inputs):
        if name is None:
            name = "chain"
        colnames = [c[0] for c in inputs.columns]
        data = [np.array(inputs.rows)]
        metadata = [{k: v[0] for k, v in inputs.meta.items()}]
        name = metadata[0].get("chain_name", "chain")
        sampler = metadata[0].get("sampler")
        final_metadata = [{k: v[0] for k, v in inputs.final_meta.items()}]
        comments = [inputs.comments[:]]
        return name, sampler, colnames, data, metadata, comments, final_metadata

    @classmethod
    def load_astropy(cls, inputs):
        name = inputs.meta.get("chain_name", "chain")
        colnames = inputs.colnames
        # convert astropy table to numpy array
        data = [np.array([inputs[c] for c in colnames]).T]
        metadata = [inputs.meta]
        sampler = metadata.get("sampler")
        final_metadata = [
            {k[6:]: v for k, v in meta.items() if k.startswith("final:")}
            for meta in metadata
        ]
        comments = [meta["comments"] for meta in metadata]
        return name, sampler, colnames, data, metadata, comments, final_metadata

    @classmethod
    def load_text_file(cls, inputs):
        colnames, data, metadata, comments, final_metadata = (
            output_module.TextColumnOutput.load_from_options({"filename": inputs})
        )
        name = metadata[0].get("chain_name", "chain")
        sampler = metadata[0].get("sampler")
        return name, sampler, colnames, data, metadata, comments, final_metadata

    @classmethod
    def load_output_object(self, inputs):
        colnames, data, metadata, comments, final_metadata = inputs.load_from_options(
            {"filename": inputs}
        )
        name = metadata[0].get("chain_name", "chain")
        sampler = metadata[0].get("sampler")
        return name, sampler, colnames, data, metadata, comments, final_metadata

    def __init_subclass__(cls):
        name = cls.__name__
        sampler = name.lower().replace("chain", "")
        cls.chain_subclasses[sampler] = cls

    def derive_extra_column(self, function):
        new_data = []
        for d in self.data:
            chain = SingleChainData(d, self.colnames)
            col, code = function(chain)
            if col is None:
                break
            # insert a new column into the chain, second from the end
            d = np.insert(d, -2, col, axis=1)
            # save the new chain
            new_data.append(d)
        if code is None:
            return

        self.colnames.insert(-2, code)
        self.data = new_data

    def derive_extra_columns(self):
        if not self.derive_file:
            return
        name = os.path.splitext(os.path.split(self.derive_file)[1])[0]
        module = import_by_path(name, self.derive_file)
        functions = [getattr(module, f) for f in dir(module) if f.startswith("derive_")]
        print(
            "Deriving new columns from these functions in {}:".format(self.derive_file)
        )
        for f in functions:
            self.derive_extra_column(f)

        # derive any additional parameters
        self.derive_extra_columns()

        # set the column names
        self.colnames = [c.lower() for c in self.colnames]
        self.data_stacked = np.concatenate(self.data).T

    def extract_ini(self, tag):
        in_ = False
        lines = []
        for line in self.comments[0]:
            line = line.strip()
            if line == "START_OF_{}_INI".format(tag):
                in_ = True
            elif line == "END_OF_{}_INI".format(tag):
                break
            elif in_:
                lines.append(line)

        s = StringIO("\n".join(lines))
        ini = Inifile(None)
        ini.read_file(s)
        return ini

    def __len__(self):
        return self.data_stacked.shape[1]

    def get_row(self, index):
        return self.data_stacked[:, index]

    def has_col(self, name):
        return name in self.colnames

    def get_col(self, index_or_name, stacked=True):
        """Get the named or numbered column."""
        if isinstance(index_or_name, int):
            index = index_or_name
        else:
            name = index_or_name
            index = self.colnames.index(name)
        cols = [d[:, index] for d in self.data]
        if stacked:
            return np.concatenate(cols)
        else:
            return cols

    @property
    def mcsamples(self):
        import getdist

        if self._mcsamples is None:
            samples = []
            names = []
            for col in self.colnames:
                if col == "weight":
                    continue
                c = self.reduced_col(col)
                samples.append(c)
                names.append(col)
            if "weight" in self.colnames:
                weights = self.reduced_col("weight")
            else:
                weights = None
            samples = np.array(samples).T
            self._mcsamples = getdist.MCSamples(
                samples=samples, weights=weights, names=names,
                name_tag=self.name
            )
        return self._mcsamples

    # Subclasses implement these
    def reduced_col(self, column):
        pass


class EmceeChain(Chain):
    def reduced_col(self, name, stacked=True):
        cols = self.get_col(name, stacked=False)
        burn = self.options.get("burn", 0)
        thin = self.options.get("thin", 1)

        if 0.0 < burn < 1.0:
            burn = int(len(cols[0]) * burn)
        else:
            burn = int(burn)

        cols = [col[burn::] for col in cols]

        if thin != 1:
            walkers = self.sampler_option("walkers")
            index = np.arange(len(cols[0]), dtype=np.int64)
            index = index // int(walkers)
            w = (index % thin) == 0
            cols = [col[w] for col in cols]

        if stacked:
            return np.concatenate(cols)
        else:
            return cols


class MetropolisChain(Chain):
    def reduced_col(self, name, stacked=True):
        cols = self.get_col(name, stacked=False)
        burn = self.options.get("burn", 0)
        thin = self.options.get("thin", 1)

        if 0.0 < burn < 1.0:
            burn = int(len(cols[0]) * burn)
        else:
            burn = int(burn)

        cols = [col[burn::thin] for col in cols]

        if stacked:
            return np.concatenate(cols)
        else:
            return np.array(cols).squeeze()


class ImportanceChain(MetropolisChain):
    # Importance sample chains may or may not
    # be based on a Metropolis-Hastings chain,
    # but they are treated the same way here.
    pass


class NestedChain(Chain):
    def reduced_col(self, name, stacked=True):
        """
        Nested sampling does not required cutting the chain
        from the main output file.

        These are also single chains, unlike the MH ones,
        so there is no stacking needed.
        """
        # stacking does not
        col = self.get_col(name)
        return np.array(col)


class PolychordChain(NestedChain):
    pass


class MultinestChain(NestedChain):
    pass


class ZeusChain(EmceeChain):
    pass


class PocoChain(MetropolisChain):
    pass


class DynestyChain(NestedChain):
    pass


class NautilusChain(NestedChain):
    pass


class SimpleListChain(Chain):
    """
    Base class for chains that are just lists of samples
    in some way. We might still want to burn or thin them though
    """

    def reduced_col(self, name, stacked=True):
        cols = self.get_col(name, stacked=False)
        thin = self.options.get("thin", 1)
        burn = self.options.get("burn", 0)
        if 0.0 < burn < 1.0:
            burn = int(len(cols[0]) * burn)
        else:
            burn = int(burn)
        cols = [col[burn::thin] for col in cols]
        return np.array(cols).squeeze()


class ListChain(SimpleListChain):
    pass


class AprioriChain(SimpleListChain):
    pass


class SingleChainData(object):
    """
    This helper object is to make it easier for users to write functions
    that derive new parameters.
    """

    def __init__(self, data, colnames):
        self.data = data
        self.colnames = colnames

    def __getitem__(self, index_or_name):
        if isinstance(index_or_name, int):
            index = index_or_name
        else:
            name = index_or_name
            index = self.colnames.index(name)
        return self.data[:, index]
