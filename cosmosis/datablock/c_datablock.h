#ifndef COSMOSIS_C_DATABLOCK_H
#define COSMOSIS_C_DATABLOCK_H

#include "datablock_status.h"
#include "datablock_types.h"
#include "section_names.h"

#ifdef __cplusplus
#include <complex> 
#include <cstdbool>
#else
#include <complex.h> 
#include <stdbool.h>
#endif

#define OPTION_SECTION "module_options"

#ifdef __cplusplus
extern "C" {
#endif

  /*
    The type c_datablock represents groups of named values.
    Values can be any of the following types:

    int, double, char array ("string"), double _Complex
    int array, double array, char array array, double _Complex array.
    multidimensional arrays of int, double, or double _Complex

    Groups of parameters are organized into named 'sections'.
  */
  typedef void c_datablock;

  /*
    make_c_datablock returns a pointer to a newly-allocated
    c_datablock object. Code in physics modules will usually not need
    to use this function; physics module functions that need to use a
    c_datablock are passed a pointer to the c_datablock object they
    are to use.

    Any c_datablock object allocated by make_c_datablock must be
    released by a matching call to destroy_c_datablock.
  */
  c_datablock*
  make_c_datablock(void);

  /*
    Deallocate all the resources associated with the given datablock.
    After this call, any use of that datablock will result in undefined
    behavior (most likely, a crash in the program).
  */
  DATABLOCK_STATUS
  destroy_c_datablock(c_datablock* s);


  c_datablock * 
  clone_c_datablock(c_datablock* s);

  /*
    Return true (1) if the datablock has a section with the given name, and
    false (0) otherwise. If either 's' or 'name' is null, return false.
  */
  bool
  c_datablock_has_section(c_datablock const* s, const char* name);

  /*
    Return the number of sections contained in the datablock. If s is
    null, return -1.
  */
  int
  c_datablock_num_sections(c_datablock const* s);


  /*
    Delete the named section.
  */
  int c_datablock_delete_section(c_datablock * s, const char * section);


  /*
    Copy the section.
    Fails and returns DBS_NAME_ALREADY_EXISTS if the destination already exists.
  */
  int c_datablock_copy_section(c_datablock * s, const char * source, const char * dest);




  /*
    Return DBS_SUCCESS if the datablock has a value in the given section
    with the given name, and an error status otherwise. The associated
    value can be of any supported type.
  */
  bool
  c_datablock_has_value(c_datablock const* s, const char* section,
			const char* name);

  /*
    Return the name of the j'th value in the named section. Return
    NULL if s is NULL, or section is NULL, or if there is no section
    with the given name, or if the named section has too few values.
  */
  const char*
  c_datablock_get_value_name(c_datablock const* s,
			     const char* section,
			     int j);

  /*
    Return the name of the j'th value of the i'th section. Return NULL
    if s is NULL, or if there are too few sections, or if there are
    too few values in the indicated section.
  */
  const char*
  c_datablock_get_value_name_by_section_index(c_datablock const* s,
					      int i, int j);


  /*
    Return the length of the named array in the given section.  If any
    of 's', 'section' or 'name' is null, or if there is no such
    section, or no such name in that section, or if the value
    associated with that name is not a 1-dimensional array, return
    -1. If the length of the array is too large to be representable as
    an int, return -2.
  */
  int
  c_datablock_get_array_length(c_datablock const* s, const char* section,
			       const char* name);

  /*
    Get the number of values in the named section.
    If block and section are non-NULL and there is a section
    with the given name then return the number of values in that
    section.

    Otherwise return -1.
  */
  int
  c_datablock_num_values(c_datablock const* s, const char* section);

  /*
    Find the dimensionality of an array with the given section and
    name.  If the object searched for is a scalar then ndim will be zero
    and an error status (DBS_WRONG_VALUE_TYPE) will be returned.
  */


  // This function is as-yet untested.
  // All I can say is that it compiles.
  // DATABLOCK_STATUS
  // c_datablock_get_ndim(c_datablock const* s, const char* section, const char * name, int * ndim);

  /*
    Return the name of the i'th section of the datablock. Note that if a
    new section is added, the ordinal position of some or all of the
    named sections may change; this is because the sections are stored
    in sorted order for speed of lookup. The caller is not intended to
    free the returned pointer; the datablock retains ownership of the
    memory buffer containing the string. A NULL pointer is returned if i
    is negative or out-of-range. Numbering of sections starts with 0.
  */


  const char*
  c_datablock_get_section_name(c_datablock const* s, int i);

  /*
    Print a report of all encountered failures to standard
    error. Return an error if 's' NULL, or if the write does not
    complete, and success otherwise.
  */
  DATABLOCK_STATUS
  c_datablock_report_failures(c_datablock* s);

  /*
    Print a report of all accesses to the datablock to standard
    output. Return an error if 's' is NULL, or if the write does not
    complete, and success otherwise.
  */
  DATABLOCK_STATUS
  c_datablock_print_log(c_datablock* s);

  DATABLOCK_STATUS
  c_datablock_log_access(c_datablock* s,
			 const char* log_type,
			 const char* section,
			 const char* name);

  /*
    Write an enumerator value into 't', corresponding to the type of
    the value stored in the given section, for the given name. Return
    DBS_SUCCESS on success, and an error status otherwise.

    If the return status is not DBS_SUCCESS, the value written into
    'val' is not defined.
  */
  DATABLOCK_STATUS
  c_datablock_get_type(c_datablock const* s,
		       const char* section,
		       const char* name,
		       datablock_type_t* val);

  /*
    The c_datablock_get_TYPE functions return DBS_SUCCESS if a value of
    type TYPE with the given name is found in the given section, and an
    error status otherwise. No conversions of type are done.

    If the return status is not DBS_SUCCESS, the value written into 'val' is not
    defined.
  */
  DATABLOCK_STATUS
  c_datablock_get_int(c_datablock* s, const char* section, const char* name, int* val);

  DATABLOCK_STATUS
  c_datablock_get_bool(c_datablock* s, const char* section, const char* name, bool* val);

  DATABLOCK_STATUS
  c_datablock_get_double(c_datablock* s, const char* section, const char* name, double* val);

  DATABLOCK_STATUS
  c_datablock_get_complex(c_datablock* s, const char* section, const char* name, double _Complex* val);

  DATABLOCK_STATUS
  c_datablock_get_string(c_datablock* s, const char* section, const char* name, char** val);

  /*
    The c_datablock_get_TYPE_default functions return DBS_SUCCESS if a
    value of type TYPE with the given name is found in the given
    section, or if no value of that name (of any type) is found in that
    section.

    If DBS_SUCCESS is returned, 'val' will be carrying either the value
    read from the c_datablock, or the user-supplied default otherwise.

    If the return status is no DBS_SUCCESS the value written into 'val'
    is not defined.
  */
  DATABLOCK_STATUS
  c_datablock_get_int_default(c_datablock* s,
			      const char* section,
			      const char* name,
			      int def,
			      int* val);

  DATABLOCK_STATUS
  c_datablock_get_bool_default(c_datablock* s,
			       const char* section,
			       const char* name,
			       bool def,
			       bool* val);

  DATABLOCK_STATUS
  c_datablock_get_double_default(c_datablock* s,
				 const char* section,
				 const char* name,
				 double def,
				 double* val);

  DATABLOCK_STATUS
  c_datablock_get_string_default(c_datablock* s,
				 const char* section,
				 const char* name,
				 const char* def,
				 char** val);

  DATABLOCK_STATUS
  c_datablock_get_complex_default(c_datablock* s,
				  const char* section,
				  const char* name,
				  double _Complex def,
				  double _Complex* val);

  /*
    The c_datablock_put_TYPE functions return DBS_SUCCESS if the given
    value was stored in the c_datablock, and an error status
    otherwise. Note that it is an error to use the
    c_datablock_put_TYPE functions to replace a value already in the
    c_datablock; use c_datablock_replace_TYPE for that.
  */
  DATABLOCK_STATUS
  c_datablock_put_int(c_datablock* s, const char* section, const char* name, int val);

  DATABLOCK_STATUS
  c_datablock_put_bool(c_datablock* s, const char* section, const char* name, bool val);

  DATABLOCK_STATUS
  c_datablock_put_double(c_datablock* s, const char* section, const char* name, double val);

  DATABLOCK_STATUS
  c_datablock_put_complex(c_datablock* s, const char* section, const char* name, double _Complex val);

  DATABLOCK_STATUS
  c_datablock_put_string(c_datablock* s, const char* section, const char* name, const char* val);

  /*
    The c_datablock_replace_TYPE functions return DBS_SUCCESS if the
    given value was stored in the c_datablock, and an error status
    otherwise. Note that it is an error to use the
    c_datablock_replace_TYPE functions to put a new value (rather than
    replacing a value) in a c_datablock. Use c_datablock_put_TYPE
    instead.
  */
  DATABLOCK_STATUS
  c_datablock_replace_int(c_datablock* s, const char* section, const char* name, int val);

  DATABLOCK_STATUS
  c_datablock_replace_bool(c_datablock* s, const char* section, const char* name, bool val);

  DATABLOCK_STATUS
  c_datablock_replace_double(c_datablock* s, const char* section, const char* name, double val);

  DATABLOCK_STATUS
  c_datablock_replace_complex(c_datablock* s, const char* section, const char* name, double _Complex val);

  DATABLOCK_STATUS
  c_datablock_replace_string(c_datablock* s, const char* section, const char* name, const char* val);

  /*
    The c_datablock_get_TYPE_array_1d functions returns DBS_SUCCESS on
    success, and and error status
    otherwise. c_datablock_get_TYPE_array_1d takes as input the
    address of an array-of-TYPE (val), into which is written the
    address of a newly-allocated array-of-TYPE; it also takes an
    address of an int (size) into which is written the size of the
    newly-allocated array.

    The user is responsible for disposing of the allocated memory (using
    'free') when it is no longer needed.
  */
  DATABLOCK_STATUS
  c_datablock_get_int_array_1d(c_datablock* s,
			       const char* section,
			       const char* name,
			       int** val,
			       int* size);

  DATABLOCK_STATUS
  c_datablock_get_double_array_1d(c_datablock* s,
				  const char* section,
				  const char* name,
				  double** val,
				  int* size);

  DATABLOCK_STATUS
  c_datablock_get_complex_array_1d(c_datablock* s,
				   const char* section,
				   const char* name,
				   double _Complex** val,
				   int* size);

  /*
    The c_datablock_get_TYPE_array_1d_preallocated functions return
    DBS_SUCCESS on success, and and error status otherwise.

    c_datablock_get_TYPE_array_1d_preallocated takes as input an
    already-allocated array-of-TYPE (val), into which is copied the
    values contained in the array. The user must specify the size of
    the pre-allocated array (maxsize). The function also takes the
    address of an int (size) into which is written the number of
    elements written into the pre-allocated array.

    If the size of the pre-allocated array smaller than the size of the
    array in the datablock, then no values are written into 'array', and
    and error status is returned. The size of the array in the datablock
    is still written into 'size' in this case.
  */
  DATABLOCK_STATUS
  c_datablock_get_int_array_1d_preallocated(c_datablock* s,
					    const char* section,
					    const char* name,
					    int* array,
					    int* size,
					    int maxsize);

  DATABLOCK_STATUS
  c_datablock_get_double_array_1d_preallocated(c_datablock* s,
					       const char* section,
					       const char* name,
					       double* array,
					       int* size,
					       int maxsize);

  DATABLOCK_STATUS
  c_datablock_get_complex_array_1d_preallocated(c_datablock* s,
						const char* section,
						const char* name,
						double _Complex* array,
						int* size,
						int maxsize);

  /*
    The c_datablock_put_TYPE_array_1d functions return DBS_SUCCESS if
    the values in the given array were stored in the c_datablock, and
    an error status otherwise. Note that it is an error use the
    c_datablock_put_TYPE_array_1d functions to replace a value already
    in the c_datablock; use c_datablock_replace_TYPE_array_1d for
    that.

    c_datablocK_put_TYPE_array_1d takes as input a pointer-to-TYPE
    'val' and an int 'sz'. 'val' is expected to be the first address
    of an array of length 'sz. Because the name of an array-of-TYPE
    can be used as a pointer-to-TYPE, one can pass the name of an
    array as 'val'.

    The values in the array are copied into the c_datablock.
  */
  DATABLOCK_STATUS
  c_datablock_put_int_array_1d(c_datablock* s,
			       const char* section,
			       const char* name,
			       int const*  val,
			       int sz);

  DATABLOCK_STATUS
  c_datablock_put_double_array_1d(c_datablock* s,
				  const char* section,
				  const char* name,
				  double const*  val,
				  int sz);

  DATABLOCK_STATUS
  c_datablock_put_complex_array_1d(c_datablock* s,
				   const char* section,
				   const char* name,
				   double _Complex const*  val,
				   int sz);

  /*
    The c_datablock_replace_TYPE_array_1d functions return DBS_SUCCESS
    if the values in the given array were stored in the c_datablock,
    and an error status otherwise. Note that it is an error use the
    c_datablock_replace_TYPE_array_1d functions to put a new value
    into the c_datablock; use c_datablock_put_TYPE_array_1d for that.

    c_datablocK_replace_TYPE_array_1d takes as input a pointer-to-TYPE
    'val' and an int 'sz'. 'val' is expected to be the first address
    of an array of length 'sz. Because the name of an array-of-TYPE
    can be used as a pointer-to-TYPE, one can pass the name of an
    array as 'val'.

    The values in the array are copied into the c_datablock.
  */
  DATABLOCK_STATUS
  c_datablock_replace_int_array_1d(c_datablock* s,
				   const char* section,
				   const char* name,
				   int const* val,
				   int sz);

  DATABLOCK_STATUS
  c_datablock_replace_double_array_1d(c_datablock* s,
				      const char* section,
				      const char* name,
				      double const* val,
				      int sz);

  DATABLOCK_STATUS
  c_datablock_replace_complex_array_1d(c_datablock* s,
				       const char* section,
				       const char* name,
				       double _Complex const* val,
				       int sz);



  DATABLOCK_STATUS
  c_datablock_get_array_ndim(c_datablock * s, 
          const char* section, 
          const char * name, 
          int * ndim);


  /*
    The c_datablock_get_TYPE_array_shape functions return DBS_SUCCESS
    if the given section has a value of the given type, and if that
    value is an array type; they return an error status otherwise.

    c_datablock_get_TYPE_array_shape takes an in 'ndims' which
    specifies the number of dimensions of the array, and an array
    'extents' that must be legnth greater than or equal to 'ndims'. If
    DBS_SUCCESS is returned, the array 'extents' will be filled with
    the extents of the n-dimensional array of the given name in the
    given section.

    A usage example is given below, in the comments on
    c_datablock_get_TYPE_array.
  */

  DATABLOCK_STATUS
  c_datablock_get_int_array_shape(c_datablock* s,
				  const char* section,
				  const char* name,
				  int ndims,
				  int* extents);

  DATABLOCK_STATUS
  c_datablock_get_double_array_shape(c_datablock* s,
				     const char* section,
				     const char* name,
				     int ndims,
				     int* extents);

  DATABLOCK_STATUS
  c_datablock_get_complex_array_shape(c_datablock* s,
				      const char* section,
				      const char* name,
				      int ndims,
				      int* extents);

  /*
    The c_datablock_get_TYPE_array functions return DBS_SUCCESS if the
    given section has a value of the given type, and that type is an
    array type, and the number of dimensions of the array is 'ndims',
    and the extent of each dimension matches the values in 'extents';
    they return an error status otherwise.

    If DBS_SUCCESS is returned, the values of the array are copied
    into 'val'.

    EXAMPLE OF USE: To get a 3-dimensional array of doubles from the
    section "sec", with name "voxels", two calls are needed. We assume
    that 's' is a c_datablock pointer. Note that we declare the array
    my_array on the stack, so that no calls to malloc or free are
    needed. If the array is too large, this will not work; see the
    comments in the code for the alternative, for large arrays.

        int extents[3];
        if (c_datablock_get_double_array_shape(s, "sec", "voxels", 3, extents) == DBS_SUCCESS)
        {
          int nx = extents[0], ny = extents[1], nz = extents[2];
          double my_array[nx][ny][nz];
          // or double* my_array = (double*)malloc(sizeof(double) * nx * ny * nz);
          if (c_datablock_get_double_array(s, "sec", "voxels", (double*)my_array, 3, extents)
               == DBS_SUCCESS)
          {
	     // use my_array here...
          }
	  // if created with malloc, you must free the memory. If created on the stack, you must
	  // not free the memory.
	  // free(my_array);
        }
  */
  DATABLOCK_STATUS
  c_datablock_get_int_array(c_datablock* s,
			    const char* section,
			    const char* name,
			    int* val,
			    int ndims,
			    int const* extents);

  DATABLOCK_STATUS
  c_datablock_get_double_array(c_datablock* s,
			       const char* section,
			       const char* name,
			       double* val,
			       int ndims,
			       int const* extents);

  DATABLOCK_STATUS
  c_datablock_get_complex_array(c_datablock* s,
				const char* section,
				const char* name,
				double _Complex* val,
				int ndims,
				int const* extents);


  /*
    The c_datablock_put_TYPE_array functions return DBS_SUCCESS if the given
    value was stored in the c_datablock, and an error status
    otherwise. Note that it is an error to use the
    c_datablock_put_TYPE_array functions to replace a value already in the
    c_datablock, use c_datablock_replace_TYPE_array (below) for that.

    Stores, under `section`, with the name `name`, an array of `ndims`
    dimensions represented by `val`. The `extents` represent the size of each
    dimension, and should be an array of `ndims` entries. Thus, a 3D array
    which would be represented in C by:

        int my_ndarr[3][2][6];

    Would be represented here as:

        int *my_ndarr = malloc(sizeof(int) * 3 * 2 * 6);
        int extents[3] = {3, 2, 6};

    Once filled with relevent values, it can be stored in the datablock as:

        int retval = c_datablock_put_int_array(datablock,
                                               "my_section", "my_name",
                                               my_ndarr, 3, extents);
        if (retval != DBS_SUCCESS) {
            ...
        }
        free(my_ndarr);

    Note that the datablock will allocate its own memory for the array; thus,
    you are responsible for freeing your own ndarray.
  */
  DATABLOCK_STATUS
  c_datablock_put_int_array(c_datablock* s,
			    const char* section,
			    const char* name,
			    int const* val,
			    int ndims,
			    int const* extents);

  DATABLOCK_STATUS
  c_datablock_put_double_array(c_datablock* s,
			       const char* section,
			       const char* name,
			       double const* val,
			       int ndims,
			       int const* extents);

  DATABLOCK_STATUS
  c_datablock_put_complex_array(c_datablock* s,
				const char* section,
				const char* name,
				double _Complex const* val,
				int ndims,
				int const* extents);

  /*
    The c_datablock_replace_TYPE_array functions return DBS_SUCCESS if the
    given value was stored in the c_datablock, and an error status
    otherwise. Note that it is an error to use the
    c_datablock_replace_TYPE_array functions to put a new value (rather than
    replacing a value) in a c_datablock. Use c_datablock_replace_TYPE_array
    instead.

    The arguments are the same pattern as for c_datablock_put_TYPE_array
    functions, see above for an explanation of the arguments. (The same memory
    management policy applies).
  */
  DATABLOCK_STATUS
  c_datablock_replace_int_array(c_datablock* s,
                            const char* section,
                            const char* name,
                            int const* val,
                            int ndims,
                            int const* extents);

  DATABLOCK_STATUS
  c_datablock_replace_double_array(c_datablock* s,
                               const char* section,
                               const char* name,
                               double const* val,
                               int ndims,
                               int const* extents);

  /*
    TODO: document these functions.
  */
  double**
  allocate_2d_double(int nx, int ny);

  void
  deallocate_2d_double(double** * z, int nx);

  DATABLOCK_STATUS
  c_datablock_put_double_grid(c_datablock* s,
            const char* section,
            const char* name_x, int n_x, double* x,
            const char* name_y, int n_y, double* y,
            const char* name_z, double** z);

  DATABLOCK_STATUS
  c_datablock_replace_double_grid(c_datablock* s,
            const char* section,
            const char* name_x, int n_x, double* x,
            const char* name_y, int n_y, double* y,
            const char* name_z, double** z);

  DATABLOCK_STATUS
  c_datablock_get_double_grid(c_datablock* s,
			      const char* section,
			      const char* name_x, int* n_x, double** x,
			      const char* name_y, int* n_y, double** y,
			      const char* name_z, double** * z);


  DATABLOCK_STATUS
  c_datablock_get_metadata(c_datablock* s, 
                         const char* section,
                         const char* name,
                         const char* key,
                         char** val
                         );


  DATABLOCK_STATUS
  c_datablock_put_metadata(c_datablock* s,
       const char* section,
       const char* name,
       const char* key,
       const char* val);


  DATABLOCK_STATUS
  c_datablock_replace_metadata(c_datablock* s,
       const char* section,
       const char* name,
       const char* key,
       const char* val);

  /*
    Serialize the datablock to a binary file at the given path.
    Returns DBS_SUCCESS on success, or DBS_LOGIC_ERROR on failure.
  */
  DATABLOCK_STATUS
  c_datablock_serialize(c_datablock const* s, const char* path);

  /*
    Deserialize a file previously written by c_datablock_serialize,
    replacing all current contents of the datablock.
    Returns DBS_SUCCESS on success, or DBS_LOGIC_ERROR on failure.
  */
  DATABLOCK_STATUS
  c_datablock_deserialize(c_datablock* s, const char* path);


#ifdef __cplusplus
}
#endif

#endif
