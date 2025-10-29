import os
import getdist
import numpy as np
from astropy.table import Table

# This isn't used directly, but has the side effect of registering the cosmosis table format:
from . import table as cosmosis_table
from . import postprocessing
from .runtime import Inifile


def cosmosis_to_getdist(filename, name, burn=0, nchain=0):
        """
        Read a cosmosis chain and convert it to a getdist MCSamples object.

        This uses the default latex.ini file bundled with cosmosis to get parameter labels,
        which covers many common parameters, but you might need to modify labels in the
        output object manually if you are using non-covered parameters.

        Parameters
        ----------
        filename : str
            The filename of the cosmosis chain file(s). If nchain>0,
            this should be the root filename without the _1.txt, _2.txt etc suffixes.
            Otherwise it should be the full filename.
        name : str
            The name to assign to the MCSamples object.
        burn : float, optional
            The fraction of samples to discard as burn-in. Default is 0.0
        nchain : int, optional
            The number of chain files to read in. If >0, the function will
            read in multiple files with suffixes _1.txt, _2.txt, etc. Default is 0.

        Returns
        -------
        getdist.MCSamples
            The MCSamples object containing the chain data.
        """
        # Get parameter labels from latex.ini that is bundled with cosmosis
        latex_file = os.path.join(os.path.dirname(postprocessing.__file__), "latex.ini")
        latex_names = Inifile(latex_file)

        # If nchain is specified, read in multiple chain files - this is usually
        # for the metropolis sampler under MPI.
        if nchain > 0:
            filenames = [filename + f"_{i}.txt" for i in range(1, nchain+1)]
            samples = []
            weights = []
            loglikes = []

            # MCSamples can accept a list of arrays when multiple chains
            # are generated, so read each chain file in turn and append
            # the data to lists.
            for filename in filenames:
                table = Table.read(filename, format='cosmosis')

                # Read the chain data for this file.
                # The metadata should be the same for all chains.
                samples_i, weights_i, names, loglikes_i, labels, sampler_type = _extract_getdist_inputs(
                    table, latex_names)
                samples.append(samples_i)
                weights.append(weights_i)
                loglikes.append(loglikes_i)
        else:
            # Otherwise if there is just a single chain file, read that in,
            # as MCSamples can also be given single arrays.
            t = Table.read(filename, format='cosmosis')
            samples, weights, names, loglikes, labels, sampler_type = _extract_getdist_inputs(t, latex_names)

        # Create and return the MCSamples object that you can use with getdist
        # as normal
        s = getdist.MCSamples(
                        samples=samples,
                        weights=weights,
                        names=names,
                        name_tag=name,
                        loglikes=loglikes,
                        labels=labels,
                        ignore_rows=burn,
                        sampler=sampler_type,
                    )

        return s




def _extract_getdist_inputs(table, latex_names):
    # Many cosmosis chains are weighted; get the weights here
    if "weight" in table.colnames:
        weights = table["weight"]
    elif "log_weight" in table.colnames:
        weights = np.exp(table["log_weight"] - np.max(table["log_weight"]))
    else:
        weights = None

    # Getdist calls the -log(posterior) column "loglikes"
    # Extract that column here if it exists, which it should in all
    # normal circumstances
    if "post" in table.colnames:
        loglikes = -table["post"]
    else:
        loglikes = None
        
    # Get the samples themselves as an array,
    # excluding the cosmosis special columns
    # Also get the parameter names.
    excluded = ["weight", "log_weight", "prior", "like", "post"]
    samples = np.array([table[col] for col in table.colnames if col not in excluded]).T
    names = [col for col in table.colnames if col not in excluded]

    # If this is an emcee chain, we need to reshape the samples
    # so that each walker is a separate chain.
    # emcee does not have a weight column so we don't have to 
    # change the shape of that.
    if table.meta["sampler"] == "emcee":
        samples = [samples[i::table.meta["walkers"]] for i in range(table.meta["walkers"])]
        loglikes = [loglikes[i::table.meta["walkers"]] for i in range(table.meta["walkers"])]

    # Get the sample labels from the latex.
    labels = names[:]
    for i, name in  enumerate(names):
        if "--" in name:
            sec, key = name.lower().split("--")
            if latex_names.has_option(sec, key):
                labels[i] = latex_names[sec, key]

    # The nested samplers in cosmosis store the logZ in the final metadata
    # (i.e. the metdata generated at the end of the run and stored at the 
    # end of the file). If this exists, we can assume this is a nested
    # sampler run, otherwise it's MCMC.
    if "final_metadata_items" in table.meta:
        if "log_z" in table.meta["final_metadata_items"]:
            sampler_type = "nested"
        else:
            sampler_type = "mcmc"

    return samples, weights, names, loglikes, labels, sampler_type
