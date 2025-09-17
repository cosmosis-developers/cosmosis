from cosmosis.datablock.cosmosis_py import DataBlock
import cosmosis.datablock.cosmosis_py.errors as errors
import numpy as np
import tempfile
import os
import yaml
import pytest

def test_string_array():
    block = DataBlock()
    # 1 input as list
    strings = "I am a mole and I live in a hole".split()
    print(strings)
    block.put_string_array_1d("my_section", "my_key", strings)
    strings2 = block.get_string_array_1d("my_section", "my_key")
    print(strings2)

    assert (strings == strings2).all()
    # 2 input as array
    block.put_string_array_1d("my_section2", "my_key2", np.array(strings))
    strings3 = block.get_string_array_1d("my_section2", "my_key2")

    assert (strings == strings3).all()
    print(strings3)

def test_string_array_logging():
    block = DataBlock()
    strings = "I am a mole and I live in a hole".split()
    print(strings)
    block.put_string_array_1d("my_section", "my_key", strings)
    nlog = block.get_log_count()
    for i in range(nlog):
        print(block.get_log_entry(i))


def test_string_array_getset():
    block = DataBlock()
    # 1 input as list
    strings = "I am a mole and I live in a hole".split()
    block["my_section", "my_key"] = strings

    strings2 = block.get_string_array_1d("my_section", "my_key")
    assert (strings == strings2).all()

    strings3 = block["my_section", "my_key"]


def test_string_array_save():
    block = DataBlock()
    # 1 input as list
    strings = "I am a mole and I live in a hole".split()
    block["my_section", "my_key"] = strings

    with tempfile.TemporaryDirectory() as dirname:
        dn = os.path.join(dirname, "dir")
        fn = os.path.join(dirname, "tar")
        yn = os.path.join(dirname, "block.yml")
        block.save_to_directory(dn)
        block.save_to_directory(fn)
        block.to_yaml(yn)

        strings2 = np.loadtxt(f"{dn}/my_section/my_key.txt", dtype=str)
        assert (strings == strings2).all()

        strings3 = yaml.safe_load(open(yn))["my_section"]["my_key"]
        assert (strings == strings3)


def test_delete():
    b = DataBlock()
    section = 'test'
    b.put_bool(section, 'b', True)
    b._delete_section(section)
    assert len(list(b.keys()))==0
    with pytest.raises(errors.BlockSectionNotFound):
        b._delete_section(section)

def test_bool():
    b = DataBlock()
    section = 'test'
    b.put_bool(section, 'b', True)
    assert b.get_bool(section, 'b')==True
    assert b.get_bool(section, 'c', default=False)==False
    assert b.get_bool(section, 'd', default=True)==True
    with pytest.raises(errors.BlockSectionNotFound):
        b.get_bool('ddd', 'c')

def test_default():
    b = DataBlock()
    section = 'test'
    assert b.get_int(section, 'a', default=14)==14
    with pytest.raises(errors.BlockSectionNotFound):
        b.get_int('test2', 'a')

    assert b.get_int(section, 'a')==14
    assert b[section, 'a'] == 14
    assert b.get_double(section, 'b', default=1.0)==1.0
    assert b.get_string(section, 'c', default="QQQ")=="QQQ"
    assert b.get_string(section, 'c', default="VVV")=="QQQ"

def test_complex():
    pass

def test_int():
    # make sure the shuffled sequence does not lose any elements
    b = DataBlock()
    section = 'test'
    b.put_int(section,'a',1)
    assert b.get_int(section,'a')==1
    with pytest.raises(errors.BlockNameNotFound):
        b.get_int('test', 'b')

    b.replace_int(section,'a',2)

    assert b.get_int(section,'a')==2
    with pytest.raises(errors.BlockNameAlreadyExists):
        b.put_int('test', 'a', 3)


def test_string():
    b = DataBlock()
    section = 'test'
    b.put_string(section, 's', 'my_string')
    assert b.get_string(section,'s')=='my_string'
    with pytest.raises(errors.BlockNameNotFound):
        b.get_string('test', 't')
    with pytest.raises(errors.BlockNameAlreadyExists):
        b.put_string('test', 's', 'my_string')


def test_int_array():
    b = DataBlock()
    section='test'
    b.put_int_array_1d(section, 'x', [1,2,3])
    r = b.get_int_array_1d(section, 'x')
    assert (r==[1,2,3]).all()

def test_double_array():
    b = DataBlock()
    section='test'
    b.put_double_array_1d(section, 'x', [1.4,2.1,3.6])
    r = b.get_double_array_1d(section, 'x')
    assert (r==[1.4,2.1,3.6]).all() 

def test_keys():
    b = DataBlock()
    section='dogs'
    b.put(section, 'x', [1.4,2.1,3.6])
    b.put(section, "n", 14)
    b.put(section, 's', 'my_string')
    section='other'
    b.put(section, 'a', 98)
    b.put(section, "b", 1.4)
    b.put_string(section, 's', 'my_string')
    keys = list(b.keys())
    assert sorted(keys) == sorted([('dogs','x'), ('dogs','n'),('dogs','s'),('other','a'),('other','b'),('other','s')])
    for k in keys:
        assert k in b

def check_blocks_equal(b1, b2):
    keys1 = list(b1.keys())
    keys2 = list(b2.keys())
    assert sorted(keys1) == sorted(keys2)
    for sec, key in keys1:
        v1 = b1.get(sec, key)
        v2 = b2.get(sec, key)
        assert np.equal(v1, v2).all()

def test_to_string():
    b = DataBlock()
    section='dogs'
    b.put(section, 'x', [1.4,2.1,3.6])
    b.put(section, "n", 14)
    b.put(section, 's', 'my_string')
    section='other'
    b.put(section, 'a', 98)
    b.put(section, "b", 1.4)
    b.put_string(section, 's', 'my_string')
    s = b.to_string()
    b2 = DataBlock.from_string(s)
    check_blocks_equal(b, b2)

def test_wrong_array_type():
    puts = {
        int:   "put_int_array_1d",
        float: "put_double_array_1d",
        str:   "put_string_array_1d",
    }
    gets = {
        int:   "get_int_array_1d",
        float: "get_double_array_1d",
        str:   "get_string_array_1d",
    }
    dtypes = list(puts.keys())

    for d1 in dtypes[:]:
        for d2 in dtypes[:]:
            if d1 is d2:
                continue

            b = DataBlock()
            section = 'section'
            key = 'key'
            put = getattr(b, puts[d1])
            get = getattr(b, gets[d2])

            value = np.array([1, 2, 3], dtype=d1)
            put(section, key, value)
            with pytest.raises(errors.BlockWrongValueType):
                get(section, key)


if __name__ == '__main__':
    # test_string_array()
    # test_string_array_save()
    test_wrong_array_type()
