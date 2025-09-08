"""
Import this module to register the cosmosis table format with astropy.

It's not quite a perfect round trip yet, but it's close enough for almost
any purpose.
"""
from cosmosis.output import utils
from astropy.table import Table
from astropy.io import registry
import os
from typing import Union, IO

def openr(f):
    if isinstance(f, str):
        return open(f, 'r')
    elif hasattr(f, 'read'):
        return f
    else:
        raise ValueError("Argument must be a filename or a readable file-like object")
def openw(f):
    if isinstance(f, str):
        return open(f, 'w')
    elif hasattr(f, 'write'):
        return f
    else:
        raise ValueError("Argument must be a filename or a file-like object")

def read_cosmosis_table(input_file: Union[str, IO, list], delimiter: str=None) -> Table:
    """
    Read a cosmosis-format text file into an astropy Table.

    This will store the metadata in the Table.meta dictionary.
    Metadata that appears after the data is stored with a special key
    "final_metadata_items" which is a list of the keys that were in
    the final metadata.

    Parameters
    ----------
    input_file : str or file-like or list
        The name of the file to read, or an open file-like object,
        or a list of lines (e.g. from f.readlines()).
    delimiter : str, optional
        The delimiter used in the file. If None (default) then
        a tab is used.

    Returns
    -------
    table : astropy.table.Table
    """
    chain = []
    metadata = {}
    final_metadata = {}
    comments = []
    started_data = False

    # This text is copied from the text_output.py load_from_options method.
    # On the major TODO list for cosmosis v4 is to refactor the I/O more generally,
    # so I won't try to avoid the copy/paste for now.
    for i, line in enumerate(openr(input_file)):
        line = line.strip()
        if not line:
            continue

        if line.startswith('#'):
            # remove the first #
            # if there is another then this is a comment,
            # not metadata
            line = line[1:]
            if i == 0:
                column_names = line.split()
            elif line.startswith('#'):
                comment = line[1:].lstrip()
                comments.append(comment)
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
                    final_metadata[key] = val
                else:
                    metadata[key] = val
        else:
            started_data = True
            words = line.split(delimiter)
            vals = [float(word) for word in words]
            chain.append(vals)

    data = Table(rows=chain, names=column_names)

    # Handle metadata. CosmoSIS distinguishes between metadata
    # that comes before the data and after the data. We store
    # both, and also store a list of which keys were in the
    # final metadata, so that writers can put them at the end
    # of the file if they want to.
    data.meta.update(metadata)
    data.meta.update(final_metadata)
    data.meta["final_metadata_items"] = list(final_metadata.keys())

    if comments:
        data.meta['comments'] = comments

    return data


def write_cosmosis_table(table: Table, filename: Union[str, IO], overwrite: bool=False, delimiter: str="\t"):
    """
    Write an astropy Table to a cosmosis-format text file.

    This will write the metadata in the Table.meta dictionary.
    Metadata that appears after the data should be listed in
    the "final_metadata_items" key in the meta dictionary.

    Parameters
    ----------
    table : astropy.table.Table
        The table to write.

    filename : str or file-like
        The name of the file to write, or an open file-like object.
    
    overwrite : bool, optional
        If True, overwrite an existing file. Default is False,
        which will raise an error if the file exists.
    
    delimiter : str, optional
        The delimiter to use in the file. Default is tab.
    """
    if (not overwrite) and os.path.exists(filename):
        raise OSError(f"File {filename} exists and overwrite=False")

    with openw(filename) as f:

        # Write the main header line.
        f.write("#" + delimiter.join(table.colnames) + "\n")

        # We will defer writing the final metadata until the end.
        final_items = table.meta.get("final_metadata_items", [])

        # Write main metadata first at the top
        for key, val in table.meta.items():
            if key == "comments" or key == "final_metadata_items":
                continue

            if key in final_items:
                continue
                
            f.write(f"#{key}={val}\n")

        # Now write any comment metadata, which is saved as
        # a double-hashed comment line.
        comments = table.meta.get("comments", [])
        for line in comments:
            f.write(f"## {line}\n")

        # Write the data itself
        for row in table:
            line = delimiter.join(str(x) for x in row) + '\n'
            f.write(line)

        # Write the final metadata at the end of the file
        for key in final_items:
            val = table.meta[key]
            f.write(f"#{key}={val}\n")

# Tell astropy about this format, so that you can write format="cosmosis"
# in astropy table read and write calls.
registry.register_reader('cosmosis', Table, read_cosmosis_table)
registry.register_writer('cosmosis', Table, write_cosmosis_table)

