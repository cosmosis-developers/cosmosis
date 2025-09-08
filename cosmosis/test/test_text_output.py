from cosmosis.output.text_output import TextColumnOutput
from cosmosis.output.cosmomc_output import CosmoMCOutput
import tempfile
import string
import numpy as np
import os
try:
    import astropy.table
    # register the cosmosis table format:
    import cosmosis.table
except ImportError:
    astropy = None

def populate_table(out, nparam, ns):
    out.metadata('NP',nparam)
    out.metadata('NS',ns)
    out.metadata('TIME','1:30pm')

    for i in range(nparam):
        p = string.ascii_uppercase[i]
        out.add_column(p, float, 'The parameter called %s'%p)

    for i in range(ns):
        x = np.arange(nparam, dtype=int)+i
        out.parameters(x)
    out.final("FINISH",True)
    out.close()

def test_text():
    with tempfile.TemporaryDirectory() as dirname:
        filename=os.path.join(dirname, 'cosmosis_temp_output_test.txt')
        ini = {'filename':filename, 'format':'text'}
        out = TextColumnOutput.from_options(ini)
        nparam = 8
        ns = 20
        populate_table(out, nparam, ns)

        #We should be able to load this table with loadtxt
        if astropy:
            t = astropy.table.Table.read(filename, format='ascii.commented_header')
            A = t['A']
            B = t['B']

            # we should also be able to read it with the new table reader
            t2 = astropy.table.Table.read(filename, format='cosmosis')

            assert np.all(t2['A'] == A)
            assert np.all(t2['B'] == B)
            assert t2.meta['NP'] == nparam
            assert t2.meta['NS'] == ns

        else:
            t = np.loadtxt(filename,dtype=int).T
            A = t[0]
            B = t[1]

        assert (A == np.arange(ns, dtype=int)).all()
        assert (B == np.arange(ns, dtype=int)+1).all()


        #or with our own method
        names, data, meta, comments, final = TextColumnOutput.load_from_options({"filename":filename})
        assert names == [string.ascii_uppercase[i] for i in range(nparam)]
        assert len(names)==nparam
        assert len(data[0])==ns
        assert meta[0]['NP']==nparam
        assert final[0]['FINISH'] is True


def test_cosmomc_output():
    with tempfile.TemporaryDirectory() as dirname:
        filename=os.path.join(dirname, 'cosmosis_temp_cosmomc_output_test.txt')

        ini = {'filename':filename, 'format':'cosmomc'}
        out = CosmoMCOutput.from_options(ini)

        nparam = 8
        ns = 20
        for i in range(nparam):
            p = string.ascii_uppercase[i]
            out.add_column(p, float, 'The parameter called %s'%p)
        out.add_column("post", float, 'The parameter called %s'%p)


        for i in range(ns):
            x = np.arange(nparam + 1, dtype=int)+i
            x[-1] = -666.0 + i
            out.parameters(x)

        # do the last one again
        out.parameters(x)


        # Need to call this to print out the last one
        out._close()


        data = np.loadtxt(filename)
        assert data.shape == (ns, nparam + 2)
        assert (data[:-1, 0] == 1).all()
        assert data[-1, 0] == 2
