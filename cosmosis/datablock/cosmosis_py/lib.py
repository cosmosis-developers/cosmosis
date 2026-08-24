import ctypes as ct
from . import dbt_types
import os

source_dir = os.path.dirname(__file__)
libcosmosis_path = os.path.join(source_dir, os.path.pardir, "libcosmosis.so")
dll = ct.cdll.LoadLibrary(libcosmosis_path)

# We export a symbol in the C code to tell us this
enum_size = ct.c_int.in_dll(dll, "cosmosis_enum_size").value

#Assuming that enums are signed ...
c_enum = {
	1:ct.c_int8,
	2:ct.c_int16,
	4:ct.c_int32,
	8:ct.c_int64,
}[enum_size]

free = dll.free
free.argtypes = [ct.c_void_p]
free.restype = None

c_block = ct.c_size_t
c_datatype = c_enum
c_status = ct.c_int
c_str = ct.c_char_p
c_int = ct.c_int
c_int_p = ct.POINTER(ct.c_int)


def load_library_function(namespace, name, argtypes, restype):
	function = getattr(dll,name)
	function.argtypes = argtypes
	function.restype = restype
	namespace[name] = function

class c_complex(ct.Structure):
	_fields_ = [("real", ct.c_double), ("imag", ct.c_double)]


def load_function_types(namespace, c_type, c_name):
	load_library_function(namespace, "c_datablock_put_%s"%c_name, [c_block, c_str, c_str, c_type], c_status)
	load_library_function(namespace, "c_datablock_replace_%s"%c_name, [c_block, c_str, c_str, c_type], c_status)
	load_library_function(namespace, "c_datablock_get_%s"%c_name, [c_block, c_str, c_str, ct.POINTER(c_type)], c_status)
	load_library_function(namespace, "c_datablock_get_%s_default"%c_name, [c_block, c_str, c_str, c_type, ct.POINTER(c_type)], c_status)

def load_array_function_types(namespace, c_type, c_name):
	load_library_function(namespace, "c_datablock_put_%s_array_1d"%c_name, [c_block, c_str, c_str, ct.POINTER(c_type), c_int], c_status)
	load_library_function(namespace, "c_datablock_replace_%s_array_1d"%c_name, [c_block, c_str, c_str, ct.POINTER(c_type), c_int], c_status)
	load_library_function(namespace, "c_datablock_get_%s_array_1d"%c_name, [c_block, c_str, c_str, ct.POINTER(ct.POINTER(c_type)), c_int_p], c_status)
	load_library_function(namespace, "c_datablock_get_%s_array_shape"%c_name, [c_block, c_str, c_str, c_int, c_int_p], c_status)
	load_library_function(namespace, "c_datablock_get_%s_array"%c_name, [c_block, c_str, c_str, ct.POINTER(c_type), c_int, c_int_p], c_status)
	load_library_function(namespace, "c_datablock_put_%s_array"%c_name, [c_block, c_str, c_str, ct.POINTER(c_type), c_int, c_int_p], c_status)
	load_library_function(namespace, "c_datablock_replace_%s_array"%c_name, [c_block, c_str, c_str, ct.POINTER(c_type), c_int, c_int_p], c_status)
	load_library_function(namespace, "c_datablock_get_%s_array_1d_preallocated"%c_name, [c_block, c_str, c_str, ct.POINTER(c_type), c_int_p, c_int], c_status)

load_function_types(locals(), ct.c_int, 'int')
load_function_types(locals(), ct.c_bool, 'bool')
load_function_types(locals(), ct.c_double, 'double')
load_function_types(locals(), c_complex, 'complex')
load_function_types(locals(), c_str, 'string')

load_array_function_types(locals(), ct.c_int, 'int')
load_array_function_types(locals(), ct.c_double, 'double')
#load_array_function_types(locals(), ct.c_complex, 'complex')

load_library_function(
	locals(), 
	"make_c_datablock",
	[],
	c_block
)

load_library_function(
       locals(), 
       "destroy_c_datablock",
       [c_block],
       c_int
)

load_library_function(
	locals(), 
	"clone_c_datablock",
	[c_block],
	c_block
)



load_library_function(
	locals(), 
	"c_datablock_num_sections",
	[c_block],
	c_int
)

load_library_function(
	locals(),
	"c_datablock_get_array_length",
	[c_block, c_str, c_str],
	c_int
	)

load_library_function(
	locals(),
	"c_datablock_get_type",
	[c_block, c_str, c_str, ct.POINTER(c_datatype)],
	c_status
	)

load_library_function(
	locals(),
	"c_datablock_has_section",
	[c_block, c_str],
	ct.c_bool
	)

load_library_function(
	locals(),
	"c_datablock_has_value",
	[c_block, c_str, c_str],
	ct.c_bool
	)

load_library_function(
	locals(),
	"c_datablock_get_section_name",
	[c_block, ct.c_int],
	c_str
	)

load_library_function(
	locals(),
	"c_datablock_get_value_name",
	[c_block, c_str, ct.c_int],
	c_str
	)

load_library_function(
	locals(),
	"c_datablock_get_value_name_by_section_index",
	[c_block, ct.c_int, ct.c_int],
	c_str
	)

load_library_function(
	locals(),
	"c_datablock_num_values",
	[c_block, c_str],
	ct.c_int
	)


load_library_function(
	locals(),
	"c_datablock_delete_section",
	[c_block, c_str],
	ct.c_int
	)

load_library_function(
	locals(),
	"c_datablock_copy_section",
	[c_block, c_str, c_str],
	ct.c_int
	)


load_library_function(
	locals(),
	"c_datablock_report_failures",
	[c_block],
	ct.c_int
	)


load_library_function(
	locals(),
	"c_datablock_print_log",
	[c_block],
	ct.c_int
	)


load_library_function(
	locals(),
	"c_datablock_log_access",
	[c_block, c_str, c_str, c_str],
	ct.c_int
	)

load_library_function(
	locals(),
	"c_datablock_get_log_count",
	[c_block],
	ct.c_int
	)

load_library_function(
	locals(),
	"c_datablock_get_log_entry",
	[c_block, ct.c_int, ct.c_int, c_str, c_str, c_str, c_str],
	c_status
	)




load_library_function(
	locals(),
	"c_datablock_put_metadata",
	[c_block, c_str, c_str, c_str, c_str],
	ct.c_int
	)

load_library_function(
	locals(),
	"c_datablock_replace_metadata",
	[c_block, c_str, c_str, c_str, c_str],
	ct.c_int
	)

load_library_function(
	locals(),
	"c_datablock_get_metadata",
	[c_block, c_str, c_str, c_str, ct.POINTER(c_str)],
	ct.c_int
	)

load_library_function(
	locals(),
	"c_datablock_put_double_grid",
	[c_block, c_str, 
	    c_str, ct.c_int, ct.POINTER(ct.c_double),
	    c_str, ct.c_int, ct.POINTER(ct.c_double),
	    c_str, ct.POINTER(ct.POINTER(ct.c_double)),
	],
	ct.c_int
	)
 

load_library_function(
	locals(),
	"c_datablock_replace_double_grid",
	[c_block, c_str, 
	    c_str, ct.c_int, ct.POINTER(ct.c_double),
	    c_str, ct.c_int, ct.POINTER(ct.c_double),
	    c_str, ct.POINTER(ct.POINTER(ct.c_double)),
	],
	ct.c_int
	)


load_library_function(
	locals(),
	"c_datablock_get_array_ndim",
	[c_block, c_str, c_str, c_int_p],
	ct.c_int
	)

load_library_function(
	locals(),
	"c_datablock_put_str_array_1d",
	[c_block, c_str, c_str, ct.POINTER(c_str), ct.c_int],
	ct.c_int
)


# Python will auto-convert returned char pointers to strings,
# but that will leak memory here, so use a void pointer instead
load_library_function(
	locals(),
	"c_datablock_get_str_array_1d_preallocated",
	[c_block, c_str, c_str, ct.POINTER(ct.c_void_p), c_int_p],
	ct.c_int
)

load_library_function(
	locals(),
	"enable_cosmosis_segfault_handler",
	[],
	None
)

load_library_function(
	locals(),
	"c_datablock_serialize",
	[c_block, c_str],
	c_status
)

load_library_function(
	locals(),
	"c_datablock_deserialize",
	[c_block, c_str],
	c_status
)
