#coding: utf-8

u"""Definition of the :class:`DataBlock` class."""

import ctypes as ct
from . import lib
from . import errors
from . import dbt_types as types
from ...utils import mkdir
from .errors import BlockError
import numpy as np
import os
import collections
import tarfile
import io
from io import StringIO, BytesIO
import sys



option_section = "module_options"
metadata_prefix = "cosmosis_metadata:"

class DataBlock(object):
	u"""A map of (section,name)->value of parameters.

	At the heart of Cosmosis is a data-containing object which is passed
	down a pipeline of processing stages, which shape and massage those
	data as they go through.  The :class:`DataBlock` class is the
	realization of this object as seen by Python modules.

	The main methods a Cosmosis module programmer is interested in given
	one of these objects are the implicitly-called `__getitem__` and
	`__setitem__`: these retrieve parameter values from the map, and put
	new ones in or replace existing ones, respectively.

	Most of the implementation detail of this class is a complete
	orthogonal set of methods which get, put and replace parameters with
	integer, boolean, string, floating-point, complex values, either as
	scalars or 1-, 2-dimensional arrays or ‘grids’, then refinement of
	these into generic :func:`get`, :func:`set` and :func:`replace`
	methods, and finally the ultimate refinement to the
	:func:`__getitem__` and :func:`__setitem__` methods themselves.

	The *grid* concept is where a two-dimensional array is flanked by two
	one-dimensional ones giving labels to the ‘rows’ and ‘columns’; these
	labels are used to address the data directly.

	"""

	GET=0
	PUT=1
	REPLACE=2
	def __init__(self, ptr=None, own=None):
		u"""Construct an empty parameter map, or possibly shadow an existing one.

		In implementation, this Python object is actually a wrapper around
		a C object.  The constructor allows for an existing object to be
		specified through the `ptr`, and then to dictate that the Python
		object is ultimately responsible for the lifetime of the
		underlying object, via the boolean-valued `own`.

		Note that it is also possible to not specify `ptr` and to specify
		`own` as `False`, in which case a new C object will be created but
		it will be left to the application to ensure proper destruction at
		the end of its lifetime.

		"""

		# Doc: Need to find out the use-case for this latter option,
		#      and if it is being used in Cosmosis now; if not, it
		#      should be removed!

		self.owns=own
		if ptr is None:
			ptr = lib.make_c_datablock()
			self.owns=True
		if own is not None:
			self.owns=own
		self._ptr = ptr
		self._as_parameter_ = ptr
	#TODO: add destructor.  destroy block if owned

	def __del__(self):
		u"""Destroy this object.

		Also destroy the underlying C object if we are deemed to own it.

		"""
		try:
			if self.owns:
				lib.destroy_c_datablock(self._ptr)
		except:
			pass

				
	def clone(self):
		u"""Make a brand-new, completely independent object, a deep copy of the existing one.

		A new object will be returned from this method which has its own
		underlying implementation, a deep copy of the parameter map we
		are holding.  This WILL entail the attempted requisition of
		enough new memory to hold the complete parameter structure.

		"""
		ptr = lib.clone_c_datablock(self._ptr)
		return DataBlock(ptr,own=True)


	@staticmethod
	def python_to_c_complex(value):
		u"""Interpret an arbitrary Python object as a lib.c_complex type.

		This convenience function will take an actual lib.c_complex
		`value` (no-op), a Python complex `value`, the first two
		components of a Python tuple `value`, or a real scalar `value` and
		return the equivalent lib.c_complex (i.e. a type which can be
		passed to a C subroutine representing a complex number).

		In the case of the scalar input, this is taken as the real part of
		the complex number and the imaginary part will be zero.

		"""
		if isinstance(value, lib.c_complex):
			return value
		elif isinstance(value, complex):
			return lib.c_complex(value.real,value.imag)
		elif isinstance(value, tuple):
			return lib.c_complex(value[0],value[1])
		else:
			return lib.c_complex(value, 0.0)

	@staticmethod
	def python_to_1d_c_array(value, numpy_type):
		u"""Create a C object equivalent to the `value` array, interpreted as `numpy_type`.

		The object will be a contiguous list—this may entail that a value
		array with strides be copied to a compressed version—of C type
		most appropriate to the representation of the Python `numpy_type`.

		"""
		value = np.array(value, dtype=numpy_type)
		#This function is for 1D arrays only
		assert value.ndim==1
		#check strides same as itemsize.
		#This may be false if e.g. the object
		#was made by looking at a slice through
		#a matrix.
		if value.itemsize != value.strides[0]:
			#If so we need to copy the data to create
			#a new object with sensible strides
			value = value.copy()
		assert value.itemsize==value.strides[0]
		#Now return pointer to start of the data
		array = np.ctypeslib.as_ctypes(value)
		array_size = value.size
		#OK, here's the difficult part.
		# We have to return the value, as well as the
		# array we have converted it to.
		# That's because the array object
		# does not maintain a pointer to the value,
		# so it is garbage collected if we don't
		# and then the array just contains junk memory
		return value, array, array_size



	def get_int(self, section, name, default=None):
		u"""Retrieve an integer value from the parameter set.

		The `name` ʼd parameter in the given `section` will be interpreted
		as an integer and returned to the caller.  If such parameter is
		not found in the map, then the `default` will be returned if it
		was given, or else a specialized :class:`BlockError` (see
		errors.py) will be thrown.  The :class:`BlockError` may also be
		thrown if a variable is found, but is not of integer type.

		"""
		r = ct.c_int()
		if default is None:
			status = lib.c_datablock_get_int(self._ptr,section.encode('ascii'),name.encode('ascii'),r)
		else:
			status = lib.c_datablock_get_int_default(self._ptr,section.encode('ascii'),name.encode('ascii'),default,r)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)
		return r.value

	def get_bool(self, section, name, default=None):
		u"""Retrieve a boolean value from the parameter set.

		The `name` parameter in the given `section` will be interpreted
		as a boolean and returned to the caller.  If such parameter is not
		found in the map, then the `default` will be returned if it was
		given, or else a specialized :class:`BlockError` (see errors.py)
		will be thrown.  The :class:`BlockError` may also be thrown if a
		variable is found, but is not of boolean type.

		"""
		r = ct.c_bool()
		if default is None:
			status = lib.c_datablock_get_bool(self._ptr,section.encode('ascii'),name.encode('ascii'),r)
		else:
			status = lib.c_datablock_get_bool_default(self._ptr,section.encode('ascii'),name.encode('ascii'),default,r)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)
		return r.value

	def get_double(self, section, name, default=None):
		u"""Retrieve a floating-point value from the parameter set.

		The `name` parameter in the given `section` will be interpreted
		as a floating-point value and returned to the caller.  If such
		parameter is not found in the map, then the `default` will be
		returned if it was given, or else a specialized
		:class:`BlockError` (see errors.py) will be thrown.  The
		:class:`BlockError` may also be thrown if a variable is found, but
		is not of floating-point type.

		"""
		r = ct.c_double()
		if default is None:
			status = lib.c_datablock_get_double(self._ptr,section.encode('ascii'),name.encode('ascii'),r)
		else:
			status = lib.c_datablock_get_double_default(self._ptr,section.encode('ascii'),name.encode('ascii'),default,r)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)
		return r.value

	def get_complex(self, section, name, default=None):
		u"""Retrieve a complex value from the parameter set.

		The `name` parameter in the given `section` will be interpreted
		as a complex value and returned to the caller.  If such parameter
		is not found in the map, then the `default` will be returned if it
		was given, or else a specialized :class:`BlockError` (see
		errors.py) will be thrown.  The :class:`BlockError` may also be
		thrown if a variable is found, but is not of complex type.

		"""
		r = lib.c_complex()
		if default is None:
			status = lib.c_datablock_get_complex(self._ptr,section.encode('ascii'),name.encode('ascii'),r)
		else:
			status = lib.c_datablock_get_complex_default(self._ptr,section.encode('ascii'),name.encode('ascii'),default,r)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)
		return r.real+1j*r.imag

	def get_string(self, section, name, default=None):
		u"""Retrieve a string value from the parameter set.

		The `name` parameter in the given `section` will be interpreted
		as a string value and returned to the caller.  If such parameter
		is not found in the map, then the `default` will be returned if it
		was given, or else a specialized :class:`BlockError` (see
		errors.py) will be thrown.  The :class:`BlockError` may also be
		thrown if a variable is found, but is not of string type.

		"""
		r = lib.c_str()
		if default is None:
			status = lib.c_datablock_get_string(self._ptr,section.encode('ascii'),name.encode('ascii'),r)
		else:
			status = lib.c_datablock_get_string_default(self._ptr,section.encode('ascii'),name.encode('ascii'),default.encode('ascii'),r)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)
		c = r.value.decode('utf-8')
		lib.free(r)
		return c

	def get_int_array_1d(self, section, name):
		u"""Retrieve an integer array from the parameter set.

		The `name` parameter in the given `section` will be understood
		as being of integer array type and returned to the caller as a
		NumPy array.  If such a parameter is not found in the map, then a
		specialized :class:`BlockError` (see errors.py) will be thrown.

		"""
		n = lib.c_datablock_get_array_length(self._ptr, section.encode('ascii'), name.encode('ascii'))
		r = np.zeros(n, dtype=np.intc)
		arr = np.ctypeslib.as_ctypes(r)
		sz = lib.c_int()
		status = lib.c_datablock_get_int_array_1d_preallocated(self._ptr, section.encode('ascii'), name.encode('ascii'), arr, ct.byref(sz), n)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)
		return r

	def get_double_array_1d(self, section, name):
		u"""Retrieve a floating-point array from the parameter set.

		The `name` parameter in the given `section` will be understood
		as being of floating-point array type and returned to the caller
		as a *NumPy* array.  If such a parameter is not found in the map,
		then a specialized :class:`BlockError` (see errors.py) will be
		thrown.

		"""
		n = lib.c_datablock_get_array_length(self._ptr, section.encode('ascii'), name.encode('ascii'))
		r = np.zeros(n, dtype=np.double)
		arr = np.ctypeslib.as_ctypes(r)
		sz = lib.c_int()
		status = lib.c_datablock_get_double_array_1d_preallocated(self._ptr, section.encode('ascii'), name.encode('ascii'), arr, ct.byref(sz), n)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)
		return r

	def get_string_array_1d(self, section, name):
		u"""Retrieve an array of strings from the datablock.

		The `name` parameter in the given `section` will be understood
		as being of 1D string array type and returned to the caller
		as a numpy array.  If such a parameter is not found in the map,
		then a specialized :class:`BlockError` (see errors.py) will be
		thrown.

		"""
		section = section.encode('ascii')
		name = name.encode('ascii')
		n = lib.c_datablock_get_array_length(self._ptr, section, name)

		# Create the array, but make it of void pointers.
		# If you use c_char_p it just comes through as an int
		# which we can free later
		array = (ct.c_void_p * n)()
		sz = lib.c_int()
		status = lib.c_datablock_get_str_array_1d_preallocated(self._ptr, section, name, array, ct.byref(sz))
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

		# Parse each element into a list.  We don't yet know that max
		# length of the strings so don't want to make a numpy array yet.
		output = []
		for i in range(n):
			# Convert to a char pointer
			c = ct.c_char_p(array[i])
			# Decode char pointer as strings
			output.append(c.value.decode('ascii'))
			# Free the pointer allocated by strdup in C
			lib.free(array[i])

		# Convert to array for output
		return np.array(output)



	def _get_array_nd(self, section, name, dtype):

		if dtype is complex or dtype is str:
			raise ValueError("Sorry - cosmosis support for 2D complex and string values is incomplete")

		ndim = lib.c_int()
		status = lib.c_datablock_get_array_ndim(self._ptr, section.encode('ascii'), name.encode('ascii'), ct.byref(ndim))
		if status:
			raise BlockError.exception_for_status(status, section, name)

		ctype, shape_function, get_function = {
			int: (ct.c_int, lib.c_datablock_get_int_array_shape, lib.c_datablock_get_int_array),
			float: (ct.c_double, lib.c_datablock_get_double_array_shape, lib.c_datablock_get_double_array),
			#complex: (lib.c_complex, lib.c_datablock_get_complex_array_shape, lib.c_datablock_get_complex_array),
			#str: (lib.c_str, lib.c_datablock_get_string_array_shape, lib.c_datablock_get_string_array),
		}[dtype]

		#Get the array extent
		extent = (ct.c_int * ndim.value)()
		status = shape_function(self._ptr, section.encode('ascii'), name.encode('ascii'), ndim, extent)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

		#Make the space for it
		N = tuple([extent[i] for i in range(ndim.value)])
		r = np.zeros(N, dtype=ctype)
		arr = r.ctypes.data_as(ct.POINTER(ctype))

		#Fill in with the data
		status = get_function(self._ptr, section.encode('ascii'), name.encode('ascii'), arr, ndim, extent)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)
		return r

	def _put_replace_array_nd(self, section, name, value, dtype, mode):
		shape = value.shape
		ndim = len(shape)
		extent = (ct.c_int * ndim)()
		for i in range(ndim): extent[i] = shape[i]
		value = value.flatten()
		p, arr, arr_size = self.python_to_1d_c_array(value, dtype)
		put_function={
			(np.intc, self.PUT):lib.c_datablock_put_int_array,
			(np.double, self.PUT):lib.c_datablock_put_double_array,
			(np.intc, self.REPLACE):lib.c_datablock_replace_int_array,
			(np.double, self.REPLACE):lib.c_datablock_replace_double_array,
		}.get((dtype,mode))
		if put_function is None:
			raise ValueError("I do not know how to save %s in %s of type %s"% (section,name, dtype))
		status = put_function(self._ptr, section.encode('ascii'), name.encode('ascii'), arr, ndim, extent)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)


	def put_double_array_nd(self, section, name, value):
		u"""Add a floating-point array parameter to the data set.

		The `value` must be an array of values which can be interpreted as
		floating-point numbers, otherwise a :class:`ValueError` will be
		raised.  If the parameter does not exist in the data set, a
		:class:`BlockError` will be raised.  The array can be any shape.

		"""
		self._put_replace_array_nd(section, name, value, np.double, self.PUT)

	def put_int_array_nd(self, section, name, value):
		u"""Add an integer array parameter to the data set.

		The value must be an array of values which can be interpreted as
		integer numbers, otherwise a :class:`ValueError` will be raised.
		If the parameter does not exist in the data set, a
		:class:`BlockError` will be raised.  The array can be any shape.

		"""
		self._put_replace_array_nd(section, name, value, np.intc, self.PUT)



	def replace_double_array_nd(self, section, name, value):
		u"""Replace a floating-point array parameter in the data set.

		The value must be an array of values which can be interpreted as
		floating-point numbers, otherwise a :class:`ValueError` will be
		raised.  If the parameter already exists in the data set, a
		:class:`BlockError` will be raised.  The new array can be any
		shape, independent of the shape of the original value in this data
		set.

		"""
		self._put_replace_array_nd(section, name, value, np.double, self.REPLACE)

	def replace_int_array_nd(self, section, name, value):
		u"""Replace an integer array parameter in the data set.

		The value must be an array of values which can be interpreted as
		integer numbers, otherwise a :class:`ValueError` will be raised.
		If the parameter already exists in the data set, a
		:class:`BlockError` will be raised.  The new array can be any
		shape, independent of the shape of the original value in this data
		set.

		"""
		self._put_replace_array_nd(section, name, value, np.intc, self.REPLACE)

	def get_double_array_nd(self, section, name):
		u"""Get a floating-point array of *a priori* unspecified shape.

		Expect :class:`BlockError` or :class:`ValueError` to be raised if
		there are extenuating circumstances.

		"""
		return self._get_array_nd(section, name, float)

	def get_int_array_nd(self, section, name):
		u"""Get an integer-valued array of *a priori* unspecified shape.

		Expect :class:`BlockError` or :class:`ValueError` to be raised if
		there are extenuating circumstances.

		"""
		return self._get_array_nd(section, name, int)

	#def get_complex_array_2d(self, section, name):
	#	return self._get_array_2d(section, name, complex)

	#def get_string_array_2d(self, section, name):
	#	return self._get_array_2d(section, name, str)

	def put_int(self, section, name, value):
		u"""Add an integer parameter to the map.

		A new parameter will be added to the current map, at (`section`,
		`name`), and will have the `value` interpreted as an integer type.
		It is an error to try to add a parameter which is already there,
		and in this case a specialized :class:`BlockError` will be raised.

		"""
		status = lib.c_datablock_put_int(self._ptr,section.encode('ascii'),name.encode('ascii'),int(value))
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def put_bool(self, section, name, value):
		u"""Add a boolean parameter to the map.

		A new parameter will be added to the current map, at (`section`,
		`name`), and will have the `value` interpreted as a boolean type.
		It is an error to try to add a parameter which is already there,
		and in this case a :class:`BlockError` will be raised.

		"""
		status = lib.c_datablock_put_bool(self._ptr,section.encode('ascii'),name.encode('ascii'),bool(value))
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def put_double(self, section, name, value):
		u"""Add a floating-point parameter to the map.

		A new parameter will be added to the current map, at (`section`,
		`name`), and will have the `value` interpreted as a floating-point
		type.  It is an error to try to add a parameter which is already
		there, and in this case a :class:`BlockError` will be raised.

		"""
		status = lib.c_datablock_put_double(self._ptr,section.encode('ascii'),name.encode('ascii'),float(value))
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def put_complex(self, section, name, value):
		u"""Add a complex parameter to the map.

		A new parameter will be added to the current map, at (`section`,
		`name`), and will have the `value` interpreted as a complex type.
		It is an error to try to add a parameter which is already there,
		and in this case a :class:`BlockError` will be raised.

		"""
		value=self.python_to_c_complex(value)
		status = lib.c_datablock_put_complex(self._ptr,section.encode('ascii'),name.encode('ascii'),value)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def put_string(self, section, name, value):
		u"""Add a string parameter to the map.

		A new parameter will be added to the current map, at (`section`,
		`name`), and will have the `value` interpreted as a string type.
		It is an error to try to add a parameter which is already there,
		and in this case a :class:`BlockError` will be raised.

		"""
		status = lib.c_datablock_put_string(self._ptr,section.encode('ascii'),name.encode('ascii'),str(value).encode('ascii'))
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def put_int_array_1d(self, section, name, value):
		u"""Add a one-dimensional integer array to the map.

		A parameter called `name` is added to `section`, and holds `value`
		interpreted as a simple array of integers.  If this interpretation
		cannot be made then a :class:`BlockError` will be raised.

		"""
		value_ref, value,n=self.python_to_1d_c_array(value, np.intc)
		status = lib.c_datablock_put_int_array_1d(self._ptr, section.encode('ascii'), name.encode('ascii'), value, n)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def put_double_array_1d(self, section, name, value):
		u"""Add a one-dimensional floating-point array to the map.

		A parameter called `name` is added to `section`, and holds `value`
		interpreted as a simple array of floating-point values.  If this
		interpretation cannot be made then a :class:`BlockError` will be
		raised.

		"""
		value_ref, value,n=self.python_to_1d_c_array(value, np.double)
		status = lib.c_datablock_put_double_array_1d(self._ptr, section.encode('ascii'), name.encode('ascii'), value, n)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def put_string_array_1d(self, section, name, value):
		u"""Add a one-dimensional floating-point array to the map.

		A parameter called `name` is added to `section`, and holds `value`
		interpreted as a simple array of floating-point values.  If this
		interpretation cannot be made then a :class:`BlockError` will be
		raised.

		"""
		array = (ct.c_char_p * len(value))()
		for i, v in enumerate(value):
			array[i] = v.encode('ascii')

		status = lib.c_datablock_put_str_array_1d(self._ptr, section.encode('ascii'), name.encode('ascii'), array, len(value))

		if status!=0:
			raise BlockError.exception_for_status(status, section, name)




	def _method_for_type(self, T, method_type):
		method={ int:    (self.get_int,     self.put_int,     self.replace_int),
				 float:  (self.get_double,  self.put_double,  self.replace_double),
				 bool:   (self.get_bool,    self.put_bool,    self.replace_bool),
				 complex:(self.get_complex, self.put_complex, self.replace_complex),
				 str:    (self.get_string,  self.put_string,  self.replace_string)
				 }.get(T)
		if method:
			return method[method_type]
		return None

	def _method_for_datatype_code(self, code, method_type):

		T={ 
			types.DBT_INT:     (self.get_int,     self.put_int,     self.replace_int),
			types.DBT_BOOL:    (self.get_bool,     self.put_bool,     self.replace_bool),
			types.DBT_DOUBLE:         (self.get_double,  self.put_double,  self.replace_double),
			types.DBT_COMPLEX: (self.get_complex, self.put_complex, self.replace_complex),
			types.DBT_STRING:  (self.get_string,  self.put_string,  self.replace_string),
			types.DBT_INT1D:   (self.get_int_array_1d,     self.put_int_array_1d,     self.replace_int_array_1d),
			types.DBT_DOUBLE1D:(self.get_double_array_1d,  self.put_double_array_1d,  self.replace_double_array_1d),
			# types.COMPLEX1D:   (self.get_complex_array_1d, self.put_complex_array_1d, self.replace_complex_array_1d),
			types.DBT_STRING1D:    (self.get_string_array_1d,  self.put_string_array_1d,  self.replace_string_array_1d),
			# types.DBT_INT2D:   (self.get_int_array_2d,     self.put_int_array_2d,     self.replace_int_array_2d)
			types.DBT_DOUBLEND:(self.get_double_array_nd,  self.put_double_array_nd, None),
			types.DBT_INTND:(self.get_int_array_nd,  self.put_int_array_nd, None),
			# types.COMPLEX2D:   (self.get_complex_array_2d, self.put_complex_array_2d, self.replace_complex_array_2d)
			# types.STRING2D:    (self.get_string_array_2d,  self.put_string_array_2d,  self.replace_string_array_2d)
				 }.get(code)
		if T is not None:
			return T[method_type]
		return None


	def _method_for_value(self, value, method_type):
		if isinstance(value, np.float32) or isinstance(value, np.float64):
			value = float(value)
		if isinstance(value, np.int32) or isinstance(value, np.int64):
			value = int(value)

		if isinstance(value, str):
			method = (self.get_string,  self.put_string,  self.replace_string)
			return method[method_type]

		T = type(value)

		method = self._method_for_type(T, method_type)
		if method: 
			return method
		if hasattr(value,'__len__'):
			#let numpy work out what type this should be.
			array = np.array(value)
			kind = array.dtype.kind
			ndim = array.ndim
			if ndim==1:
				#1D arrays have their own specific methods,
				#for integer and float
				if kind=='i':
					method = (self.get_int_array_1d,self.put_int_array_1d,self.replace_int_array_1d)
				elif kind=='f':
					method = (self.get_double_array_1d,self.put_double_array_1d,self.replace_double_array_1d)
				elif kind == 'U':
					method = (self.get_string_array_1d,self.put_string_array_1d,self.replace_string_array_1d)
			#otherwise we just use the generic n-d arrays
			elif kind=='i':
				method = (self.get_int_array_nd,self.put_int_array_nd,self.replace_int_array_nd)
			elif kind=='f':
				method = (self.get_double_array_nd,self.put_double_array_nd,self.replace_double_array_nd)
			if method:
				return method[method_type]
		raise ValueError("I do not know how to handle this type %r %r"%(value,type(value)))
	
	def get(self, section, name):
		u"""Get the value of parameter with `name` in `section`.

		The type value returned from this method will reflect the type of
		value held in the underlying map implementation.  In circumstances
		where this either cannot be ascertained or cannot be converted
		simply to a native Python type, then either a :class:`BlockError`
		or :class:`ValueError` will be raised.

		"""
		type_code_c = lib.c_datatype()
		status = lib.c_datablock_get_type(self._ptr, section.encode('ascii'), name.encode('ascii'), ct.byref(type_code_c))
		if status:
			raise BlockError.exception_for_status(status, section, name)
		type_code = type_code_c.value
		method = self._method_for_datatype_code(type_code,self.GET)
		if method:
			return method(section,name)
		raise ValueError("Cosmosis internal error; unknown type of data. section: %s, name: %s, type_code: %s" % (section, name, type_code))

	def put(self, section, name, value, **meta):
		u"""Add a parameter with `value` at (`section`, `name`) in the map.

		The parameter stored in the map will have a type which
		reflects the type of `value`.

		If provided, `meta` should be a map of key/value pairs, and
		these will be appended to the inserted parameter as meta-data,
		converted to string type.

		It is an error to insert a parameter when there already is an
		entry at (`section`, `name`), in which case a :class:`BlockError`
		specialization will be raised.

		"""
		method = self._method_for_value(value,self.PUT)
		method(section, name, value)
		for (key, val) in list(meta.items()):
			self.put_metadata(section, name, str(key), str(val))

	def replace(self, section, name, value):
		u"""Replace the value of a parameter at (`section`, `name`) in the map with `value`.

		The parameter newly stored in the map will have a type which
		reflects the type of `value`.

		It is an error to attempt to replace a parameter not already
		present in the map, in which case a :class:`BlockError`
		specialization will be raised.

		"""
		method = self._method_for_value(value,self.REPLACE)
		method(section, name, value)


	def replace_int(self, section, name, value):
		u"""Change the value of an integer parameter in the map.

		The parameter at (`section`, `name`) will be given the new
		`value`.  It is an error to attempt to replace a value which is
		not already in the map, and a :class:`BlockError` will be raised
		in this case.

		"""
		status = lib.c_datablock_replace_int(self._ptr,section.encode('ascii'),name.encode('ascii'),value)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def replace_bool(self, section, name, value):
		u"""Change the value of a boolean parameter in the map.

		The parameter at (`section`, `name`) will be given the new
		`value`.  It is an error to attempt to replace a value which is
		not already in the map, and a :class:`BlockError` will be raised
		in this case.

		"""
		status = lib.c_datablock_replace_bool(self._ptr,section.encode('ascii'),name.encode('ascii'),value)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def replace_double(self, section, name, value):
		u"""Change the value of a floating-point parameter in the map.

		The parameter at (`section`, `name`) will be given the new
		`value`.  It is an error to attempt to replace a value which is
		not already in the map, and a :class:`BlockError` will be raised
		in this case.

		"""
		r = ct.c_double()
		status = lib.c_datablock_replace_double(self._ptr,section.encode('ascii'),name.encode('ascii'),value)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def replace_complex(self, section, name, value):
		u"""Change the value of a complex parameter in the map.

		The parameter at (`section`, `name`) will be given the new
		`value`.  It is an error to attempt to replace a value which is
		not already in the map, and a :class:`BlockError` will be raised
		in this case.

		"""
		value=self.python_to_c_complex(value)
		status = lib.c_datablock_replace_complex(self._ptr,section.encode('ascii'),name.encode('ascii'),value)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def replace_string(self, section, name, value):
		u"""Change the value of a string parameter in the map.

		The parameter at (`section`, `name`) will be given the new
		`value`.  It is an error to attempt to replace a value which is
		not already in the map, and a :class:`BlockError` will be raised
		in this case.

		"""
		status = lib.c_datablock_replace_string(self._ptr,section.encode('ascii'),name.encode('ascii'),str(value).encode('ascii'))
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def replace_int_array_1d(self, section, name, value):
		u"""Replace the value of a parameter with a simple integer array.

		The parameter at (`section`, `name`) is replaced with `value`,
		interpreted as a one-dimensional array.

		If this cannot be done then a :class:`BlockError` specialization
		will be raised.

		"""
		value_ref, value,n=self.python_to_1d_c_array(value, np.intc)
		status = lib.c_datablock_replace_int_array_1d(self._ptr, section.encode('ascii'), name.encode('ascii'), value, n)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def replace_double_array_1d(self, section, name, value):
		u"""Replace the value of a parameter with a simple floating-point array.

		The parameter at (`section`, `name`) is replaced with `value`,
		interpreted as a one-dimensional array.

		If this cannot be done then a :class:`BlockError` specialization
		will be raised.

		"""
		value_ref, value,n=self.python_to_1d_c_array(value, np.double)
		status = lib.c_datablock_replace_double_array_1d(self._ptr, section.encode('ascii'), name.encode('ascii'), value, n)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def replace_string_array_1d(self, section, name, value):
		u"""Replacing string arrays is not yet implemented

		"""
		raise NotImplementedError("CosmoSIS cannot yet replace string arrays. "
								  "Please open an issue if you need this feature")


	def has_section(self, section):
		u"""Indicate whether or not there is a given `section` in the data set.

		The `section` should be a string holding the name of the section.

		"""
		has = lib.c_datablock_has_section(self._ptr, section.encode('ascii'))
		return bool(has)

	def has_value(self, section, name):
		u"""Indicate whether or not a parameter is in the map.

		Both `section` and `name` should be strings.

		"""
		has = lib.c_datablock_has_value(self._ptr, section.encode('ascii'), name.encode('ascii'))
		return bool(has)

	def __getitem__(self, section_name):
		u"""Get the value of a parameter with `section`, name in the tuple section_name.

		Implicit use of this method is the recommended way to get the
		value of a parameter in the map.  If the argument cannot be
		interpreted as at least a two-item tuple then a
		:class:`ValueError` will be raised.

		"""
		try:
			(section,name) = section_name
		except ValueError:
			raise ValueError("You must specify both a section and a name to get or set a block item: b['section','name']")
		return self.get(section, name)

	def __setitem__(self, section_name, value):
		u"""Set a parameter with value in the map.

		The section_name must be a tuple with the parameterʼs section and
		name as the first two items (else a :class:`ValueError` will be
		raised).  Implicit use of this method is the recommended way to
		both insert parameters into the map, or to introduce new ones.

		"""

		try:
			(section,name) = section_name
		except ValueError:
			raise ValueError("You must specify both a section and a name to get or set a block item: b['section','name']")
		if self.has_value(section, name):
			self.replace(section, name, value)
		else:
			self.put(section, name, value)

	def __contains__(self, section_name):
		u"""Indicate whether there is a parameter with given section/name in the database.

		The section and name must be specified as the first two items of a
		tuple, or else a :class:`ValueError` will be raised.  Normally a
		boolean is returned to indicate prescence of said parameter.
		Implicit use of this method is the recommended way to determine if
		a parameter is present in the data set.

		"""
		if isinstance(section_name, str):
			return self.has_section(section_name)
		try:
			(section,name) = section_name
		except ValueError:
			raise ValueError("You must specify both a section and a name to get or set a block item: b['section','name']")
		return self.has_value(section, name)

	def sections(self):
		u"""Return a list of strings with the names of all sections in the data set.

		"""
		n = lib.c_datablock_num_sections(self._ptr)
		return [lib.c_datablock_get_section_name(self._ptr, i).decode('utf-8') for i in range(n)]


	def keys(self, section=None):
		u"""Return all keys in the collection, or, if `section` is specified, all keys under that section.

		If `section` is specified, it must be a string naming a section
		for whose keys are requested.

		In all cases a list of pairs of strings will be returned, the
		elements of each being the `section` and name of each parameter.

		"""
		if section is None:
			sections = self.sections()
		else:
			sections = [section]
		keys = []
		for section in sections:
			n_value = lib.c_datablock_num_values(self._ptr, section.encode('ascii'))
			for i in range(n_value):
				name = lib.c_datablock_get_value_name(self._ptr, section.encode('ascii'), i).decode('utf-8')
				keys.append((section,name))
		return keys


	def _delete_section(self, section):
		"Internal use only!"
		status = lib.c_datablock_delete_section(self._ptr, section.encode('ascii'))
		if status!=0:
			raise BlockError.exception_for_status(status, section, "<tried to delete>")

	def _copy_section(self, source, dest):
		"Internal use only!"
		status = lib.c_datablock_copy_section(self._ptr, source.encode('ascii'), dest.encode('ascii'))
		if status!=0:
			raise BlockError.exception_for_status(status, dest, "<tried to copy>")


	@staticmethod
	def _parse_metadata_key(key):
		key = key[len(metadata_prefix):].strip(":")
		s = key.index(":")
		if s==-1:
			raise ValueError("Could not understand metadata")
		name = key[:s]
		meta = key[s+1:]
		return name, meta

	def save_to_file(self, dirname, clobber=False):
		u"""Effectively :func:`save_to_directory` with the result tarʼd and compressed to a single file.

		The `dirname` argument here is actually a file name without an
		extension; the path to the file will be created in the file system
		if necessary (:class:`ValueError` will be raised if this cannot be
		accomplished), and “.tgz” will be appended to the file name.

		"""
		filename = dirname + ".tgz"

		base_dirname,base_filename=os.path.split(filename)
		if base_dirname:
			try:
				mkdir(base_dirname)
			except OSError:
				pass

		if os.path.exists(filename) and not clobber:
			raise ValueError("File %s already exists and not clobbering"%filename)

		tar = tarfile.open(filename, "w:gz")

		for (section, scalar_outputs, vector_outputs, meta) in self._save_paths():
			#Save all the vector outputs as individual files
			for name, value in vector_outputs:
				vector_outfile = os.path.join(dirname,section,name+'.txt')
				header = "%s\n"%name
				if value.ndim>2:
					header += "shape = %s\n"%str(value.shape)
					print("Flattening %s--%s when saving; shape info in header" % (section,name))
					value = value.flatten()
				if name in meta:
					for key,val in list(meta[name].items()):
						header+='%s = %s\n' % (key,val)

				#Save this file into the tar file
				string_output = io.BytesIO()
				if value.dtype.kind == "U":
					np.savetxt(string_output, value, header=header.rstrip("\n"), fmt="%s")
				else:
					np.savetxt(string_output, value, header=header.rstrip("\n"))
				string_output.seek(0)
				info = tarfile.TarInfo(name=vector_outfile)
				info.size=len(string_output.getvalue())
				tar.addfile(tarinfo=info, fileobj=string_output)

			#Save all the scalar outputs together as a single file
			#inside the tar file
			if scalar_outputs:
				scalar_outfile = os.path.join(dirname,section,"values.txt")
				string_output = io.BytesIO()
				for s in scalar_outputs:
					line = "{} = {}\n".format(s[0], s[1])
					string_output.write(line.encode())
					if s[0] in meta:
						for key,val in list(meta[s[0]].items()):
							line = "#{} {} = {}\n".format(s[0], key, val)
							string_output.write(line.encode())
				string_output.seek(0)
				info = tarfile.TarInfo(name=scalar_outfile)
				info.size=len(string_output.getvalue())
				tar.addfile(tarinfo=info, fileobj=string_output)
		tar.close()


	def save_to_directory(self, dirname, clobber=False):
		u"""Save the entire contents of this parameter map in the filesystem under `dirname`.

		The data are all written out long-hand in ASCII.  Each unique
		section will go to its own sub-directory, in which all the
		scalar parameters in that section go into a single file
		(‘values.txt’), and of the ‘composite’ data each go into their
		own file, named after the parameter key.

		The path, including `dirname`, will be created if necessary.

		"""
		try:
			mkdir(dirname)
		except OSError:
			if not clobber:
				print("Not clobbering", clobber)
				raise
		for (section, scalar_outputs, vector_outputs, meta) in self._save_paths():
			
			#Create the sub-directory for this 
			#section
			try:
				mkdir(os.path.join(dirname,section))
			except OSError:
				if not clobber:
					raise

			#Save all the vector outputs as individual files
			for name, value in vector_outputs:
				vector_outfile = os.path.join(dirname,section,name+'.txt')
				header = "%s\n"%name
				if value.ndim>2:
					header += "shape = %s\n"%str(value.shape)
					print("Flattening %s--%s when saving; shape info in header" % (section,name))
					value = value.flatten()
				if name in meta:
					for key,val in list(meta[name].items()):
						header+='%s = %s\n' % (key,val)
				if value.dtype.kind == "U":
					np.savetxt(vector_outfile, value, header=header.rstrip("\n"), fmt='%s')
				else:
					np.savetxt(vector_outfile, value, header=header.rstrip("\n"))

			#Save all the scalar outputs together as a single file
			if scalar_outputs:
				f=open(os.path.join(dirname,section,"values.txt"), 'w')
				for s in scalar_outputs:
					f.write("%s = %r\n"%s)
					if s[0] in meta:
						for key,val in list(meta[s[0]].items()):
							f.write("#%s %s = %s\n"%(s[0],key,val))
				f.close()

	def _save_paths(self):
		keys = list(self.keys())
		sections = set(k[0] for k in keys)
		for section in sections:
			scalar_outputs = []
			meta = collections.defaultdict(dict)
			vector_outputs = []
			for k in keys:
				sec, name = k
				if sec!=section: continue
				if name.startswith(metadata_prefix):
					target, metakey = self._parse_metadata_key(name)
					meta[target][metakey] = self[section,name]
					continue
				value = self[section,name]
				if np.isscalar(value):
					scalar_outputs.append((name,value))
				else:
					vector_outputs.append((name,value))
			yield section, scalar_outputs, vector_outputs, meta


	def report_failures(self):
		u"""Dump a human-readable list of failed-action log entries to the standard error channel.

		The entries appear one per line, with space-separated items
		corresponding to the verb, section and name, and data-type of the
		parameter.

		"""
		status = lib.c_datablock_report_failures(self._ptr)
		if status!=0:
			raise BlockError.exception_for_status(status, "", "")

	def print_log(self):
		u"""Dump a human-readable list of log entries to standard output.

		The entries appear one per line, with space-separated items
		corresponding to the verb, section and name, and data-type of the
		parameter.

		"""
		status = lib.c_datablock_print_log(self._ptr)
		if status!=0:
			raise BlockError.exception_for_status(status, "", "")

	def get_log_count(self):
		u"""Return the number of entries in the log."""
		return lib.c_datablock_get_log_count(self._ptr)

	def get_log_entry(self, i):
		u"""Get the iʼth log entry.

		The return is a tuple of four strings indicating the verb (i.e.,
		logged action), section and name of the parameter, and the data
		type held by the parameter.

		"""
		smax = 128
		ptype = ct.create_string_buffer(smax)
		section = ct.create_string_buffer(smax)
		name = ct.create_string_buffer(smax)
		dtype = ct.create_string_buffer(smax)
		status = lib.c_datablock_get_log_entry(self._ptr, i, smax, ptype, section, name, dtype)
		if status:
			raise ValueError("Asked for log entry above maximum or less than zero")
		return ptype.value.decode('utf-8'), section.value.decode('utf-8'), name.value.decode('utf-8'), dtype.value.decode('utf-8')

	def log_access(self, log_type, section, name):
		u"""Add an entry to the end of this :class:`DataBlock` access log.

		The `log_type` describes the action performed on the parameter at
		(`section`, `name`).  It should be one of the strings displayed in
		*datablock_logging.cc*, viz: "READ-OK", "WRITE-OK", "READ-FAIL",
		"WRITE-FAIL", "READ-DEFAULT", "REPLACE-OK", "REPLACE-FAIL",
		"CLEAR", "DELETE", or "MODULE-START".

		"""
		status = lib.c_datablock_log_access(self._ptr, log_type.encode('ascii'), section.encode('ascii'), name.encode('ascii'))
		if status!=0:
			raise BlockError.exception_for_status(status, "", "")

	def get_metadata(self, section, name, key):
		u"""Get the metadata called `key` attached to parameter `name` under `section`.

		If the data do not exist at the requested address, then a
		:class:`BlockError` will be raised.

		"""
		r = lib.c_str()
		status = lib.c_datablock_get_metadata(self._ptr,section.encode('ascii'),name.encode('ascii'),key.encode('ascii'), r)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)
		return r.value.decode('utf-8')

	def put_metadata(self, section, name, key, value):
		u"""Associate `value` with the meta-`key` attached to parameter `name` under `section`.

		If there is no parameter under (`section`, `name`) then a
		:class:`BlockError` will be raised.

		"""
		status = lib.c_datablock_put_metadata(self._ptr,section.encode('ascii'),name.encode('ascii'),key.encode('ascii'), value.encode('ascii'))
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def replace_metadata(self, section, name, key, value):
		u"""Associate `value` with the meta-`key` attached to parameter `name` under `section`.

		If there is no parameter under (`section`, `name`) then a
		:class:`BlockError` will be raised.

		"""
		status = lib.c_datablock_replace_metadata(self._ptr,section.encode('ascii'),name.encode('ascii'),key.encode('ascii'), value.encode('ascii'))
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def put_grid(self, section, name_x, x, name_y, y, name_z, z):
		u"""Put a grid into the map.

		The grid is put into `section`, using keys `name_x`, `name_y` and
		`name_z` to locate the data.  The data comprise the array `x`
		holding a set of ‘labels’ for the x-axis, an array `y` holding
		labels for the y-axis, and then a two-dimensional array`z`, whose
		sizes must correspond with the `x`- and `y`-sizes, which holds the
		actual data inside the grid.

		If there are any problems, most notably with the sizes of the
		arrays not being compatible, then a :class:`ValueError` will be
		raised.

		"""
		self._grid_put_replace(section, name_x, x, name_y, y, name_z, z, False)

	def get_grid(self, section, name_x, name_y, name_z):
		u"""Return a triple of arrays, representing a grid of data.

		The strings `name_x`, `name_y` and `name_z` must be keys under
		`section` which index data making up a grid; they must be the same
		set used in a call to :func:`replace_grid` or :func:`put_grid`
		used to establish the grid in the first place (except that the x-
		and y-axes are allowed to be transposed).

		The return is a triple of arrays: the first two elements hold the
		labels along the axes and the third element is a two-dimensional
		array holding the data deemed to be inside the grid itself.

		If the `name_*`ʼs do not correspond correctly with those of an
		established grid then a :class:`BlockError` will be raised.

		"""
		name_x = name_x.lower()
		name_y = name_y.lower()
		name_z = name_z.lower()
		x = self[section, name_x]
		y = self[section, name_y]
		z = self[section, name_z]
		sentinel_key = "_cosmosis_order_%s"%name_z
		sentinel_value = self[section, sentinel_key].lower()

		if sentinel_value== "%s_cosmosis_order_%s" % (name_x, name_y):
			assert z.shape==(x.size, y.size)
			return x, y, z
		elif sentinel_value== "%s_cosmosis_order_%s" % (name_y, name_x):
			assert z.T.shape==(x.size, y.size)
			return x, y, z.T
		else:
			raise BlockError.exception_for_status(errors.DBS_WRONG_VALUE_TYPE, section, name_z)



	def replace_grid(self, section, name_x, x, name_y, y, name_z, z):
		u"""Put a grid into the map.

		The grid is put into `section`, using keys `name_x`, `name_y` and
		`name_z` to locate the data.  The data comprise the array `x`
		holding a set of ‘labels’ for the x-axis, an array `y` holding
		labels for the y-axis, and then a two-dimensional array`z`, whose
		sizes must correspond with the `x`- and `y`-sizes, which holds the
		actual data inside the grid.

		If there are any problems, most notably with the sizes of the
		arrays not being compatible, then a :class:`ValueError` will be
		raised.

		"""
		self._grid_put_replace(section, name_x, x, name_y, y, name_z, z, True)

	def _grid_put_replace(self, section, name_x, x, name_y, y, name_z, z, replace):
		# These conversions do not create new objects if x,y,z are already arrays.
		x = np.asarray(x)
		y = np.asarray(y)
		z = np.asarray(z)

		if x.ndim!=1 or y.ndim!=1 or z.ndim!=2 or z.shape!=(x.size,y.size):
			msg = """
	Your code tried to save or replace a grid {name_z}[{name_x}, {name_y}] in section {}. 
	This requires 1D {name_x}, 1D {name_y}, 2D {name_z} and the shape of {name_z} to be (len({name_x}),len({name_y})).
	Whereas your code tried:
	{name_x} ndim = {}    [{}]
	{name_y} ndim = {}    [{}]
	{name_z} ndim = {}    [{}]
	{name_x} shape = {}
	{name_y} shape = {}
	{name_z} shape = {}   [{}]
			""".format(section, 
				x.ndim, "OK" if x.ndim==1 else "WRONG",
				y.ndim, "OK" if y.ndim==1 else "WRONG",
				z.ndim, "OK" if z.ndim==2 else "WRONG",
				x.shape,
				y.shape,
				z.shape, "OK" if z.shape==(x.size, y.size) else "WRONG",
				name_z=name_z, name_x=name_x, name_y=name_y
				)
			raise ValueError(msg)

		self[section, name_x] = x
		self[section, name_y] = y
		self[section, name_z] = z

		sentinel_key = "_cosmosis_order_%s"%name_z
		sentinel_value = "%s_cosmosis_order_%s" % (name_x, name_y)
		self[section, sentinel_key] = sentinel_value.lower()

	def get_first_parameter_use(self, params_of_interest):
		u"""Analyze the log and figure out when each parameter is first used"""
		params_by_module = collections.OrderedDict()
		current_module = []
		#make a copy of the parameter list so we can remove things
		#from it as we find their first use
		params = [(p.section,p.name) for p in params_of_interest]
		#now actually parse the log
		current_module = None
		current_name = "None"
		for i in range(self.get_log_count()):
			ptype, section, name, _ = self.get_log_entry(i)
			if ptype=="MODULE-START":
				# The previous current_module is already the
				#last element in params_by_module (unless it's the
				#very first one in which case we discard it because
				#it is the parameters being set in the sampler)
				current_module = []
				current_name = section
				params_by_module[current_name] = current_module
			elif ptype=="READ-OK" and (section,name) in params:
				current_module.append((section,name))
				params.remove((section,name))
		#Return a list of lists of parameter first used in each section
		return params_by_module

	@classmethod
	def from_yaml(cls, filename_or_stream):
		import yaml

		# work around problem with py2 yaml on unicode
		# https://stackoverflow.com/questions/27518976/how-can-i-get-pyyaml-safe-load-to-handle-python-unicode-tag
		if sys.version_info[0]==2:
			def constructor(loader, node):
				return node.value
			yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/unicode", constructor)

		block = cls()
		if isinstance(filename_or_stream, str):
			stream = open(filename_or_stream)
		else:
			stream = filename_or_stream

		data = yaml.safe_load(stream)

		if not isinstance(data, dict) or (not all(isinstance(k, str) for k in data.keys())) or (not all(isinstance(v, dict) for v in data.values())):
			raise ValueError("Wrong format yaml file {}".format(filename_or_stream))

		for section, contents in data.items():
			for key, value in contents.items():
				if isinstance(value, list):
					value = np.array(value)
				block[section, key] = value

		return block

	def to_yaml(self, filename_or_stream):
		import yaml
		if isinstance(filename_or_stream, str):
			stream = open(filename_or_stream, 'w')
		else:
			stream = filename_or_stream

		data = {}
		for section in self.sections():
			data[section] = {}
			for (_, key) in self.keys(section):
				value = self[section, key]
				if isinstance(value, np.ndarray):
					value = value.tolist()
				data[section][key] = value

		yaml.dump(data, stream)

	@classmethod
	def from_pickle(cls, filename):
		import pickle

		block = cls()
		with open(filename + ".pkl", 'rb') as f:
			data = pickle.load(f)
		for section, values in data.items():
			for key, value in values.items():
				block[section, key] = value
		return block

	def to_pickle(self, filename):
		import pickle

		data = {}
		for section in self.sections():
			data[section] = {}
			for (_, key) in self.keys(section):
				value = self[section, key]
				if isinstance(value, np.ndarray):
					value = value.tolist()
				data[section][key] = value

		with open(filename + ".pkl", 'wb') as f:
			pickle.dump(data, f)

	@classmethod
	def from_dict(cls, d):
		b = cls()
		for section, values in d.items():
			for key, value in values.items():
				b[section, key] = value
		return b

	@classmethod
	def from_string(cls, s):
		if sys.version_info[0]==2:
			sio = BytesIO(s)
		else:
			sio = StringIO(s)
		return cls.from_yaml(sio)

	def to_string(self):
		if sys.version_info[0]==2:
			sio = BytesIO()
		else:
			sio = StringIO()
		self.to_yaml(sio)
		sio.seek(0)
		return sio.read()

	def __reduce__(self):
		return (datablock_from_string, (self.to_string(),))


# This is not needed under python 3, where, the __reduce__ method
# above can return the class method, but it is under python 2.
def datablock_from_string(s):
	return DataBlock.from_string(s)






class SectionOptions(object):
	"""
	The SectionOptions object wraps is a handy short-cut to let you
	look up objects in a DataBlock object, but looking specifically at
	the special section in an ini file that refers to "the section that 
	defines the current module"

	"""
	def __init__(self, block):
		self.block=block

	def has_value(self, name):
		has = self.block.has_value(option_section, name)
		return bool(has)



def _make_getter(cls, name):
	if name=='__getitem__':
		def getter(self, key):
			return self.block[option_section, key]
	elif "array" in name:
		def getter(self, key):
			return getattr(self.block, name)(option_section, key)
	else:
		def getter(self, key, default=None):
			return getattr(self.block, name)(option_section, key, default=default)

	return getter




for name in dir(DataBlock):
	if name.startswith('get') or name=='__getitem__':
		setattr(SectionOptions, name, _make_getter(SectionOptions, name))
