#include <assert.h>
#include <complex.h> // the C header
#include <complex>   // the C++ header
#include <string.h>
#include <iostream>
#include <functional>
#include <numeric>

#include "datablock.hh"
#include "section.hh"
#include "entry.hh"
#include "c_datablock.h"
#include "ndarray.hh"
#include "clamp.hh"

using cosmosis::DataBlock;
using cosmosis::Section;
using cosmosis::Entry;
using cosmosis::complex_t;
using cosmosis::ndarray;
using cosmosis::clamp;
using std::string;
using std::vector;
using std::cout;
using std::cerr;
using std::endl;

namespace
{
  // clang seems to have different support for C-language _Complex
  // than does gcc, so we have the following conversion functions
  // that work under both compilers. They rely on the fact that both
  // C "double _Complex" and C++ "std::complex<double>" have the same
  // layout as double[2].
  std::complex<double> from_Complex(double _Complex z)
  {
    double const* p = reinterpret_cast<double*>(&z);
    return {p[0], p[1]};
  }
  
  double _Complex from_complex(std::complex<double> z)
  {
    double _Complex res;
    double tmp[] = { z.real(), z.imag()};
    res = *reinterpret_cast<double _Complex*>(tmp);  
    return res;
  }
  
  std::vector<std::complex<double>> from_Complex(double _Complex const* first, std::size_t sz)
  {
    std::vector<std::complex<double>> res(sz);
    for (std::size_t i = 0; i != sz; ++i)
    {
      res[i] = from_Complex(first[i]);
    }
    return res;
  }
}

extern "C"
{
  // This seems to be the appropriate incantation to export this
  extern const int cosmosis_enum_size = sizeof(datablock_type_t);

  c_datablock* make_c_datablock(void)
  {
    return new cosmosis::DataBlock();
  }

  c_datablock * 
  clone_c_datablock(c_datablock* s){
    DataBlock const* p = static_cast<DataBlock const*>(s);
    return new cosmosis::DataBlock(*p);
  }


  bool c_datablock_has_section(c_datablock const* s, const char* name)
  {
    if (s == nullptr || name == nullptr) return false;
    DataBlock const* p = static_cast<DataBlock const*>(s);
    return p->has_section(name);
  }

  int c_datablock_delete_section(c_datablock * s, const char * section)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    auto p = static_cast<DataBlock *>(s);
    return p->delete_section(section);
  }

  int c_datablock_copy_section(c_datablock * s, const char * source, const char * dest)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (source == nullptr) return DBS_SECTION_NULL;
    if (dest == nullptr) return DBS_SECTION_NULL;
    auto p = static_cast<DataBlock *>(s);
    return p->copy_section(source, dest);
  }


  int c_datablock_num_sections(c_datablock const* s)
  {
    if (s == nullptr) return -1;
    auto p = static_cast<DataBlock const*>(s);
    return p->num_sections();
  }

  int c_datablock_num_values(
    c_datablock const* s, const char* section)
  {
    if (s == nullptr) return -1;
    if (section == nullptr) return -1;
    DataBlock const* p = static_cast<DataBlock const*>(s);
    return clamp(p->num_values(section));
  }

  bool c_datablock_has_value(c_datablock const* s,
                                         const char* section,
                                         const char* name)
  {
    if (s == nullptr) return false;
    if (section == nullptr) return false;
    if (s == nullptr) return false;
    if (section == nullptr) return false;
    if (name == nullptr) return false;
    DataBlock const* p = static_cast<DataBlock const*>(s);
    return p->has_val(section, name);
  }

    // This function is as-yet untested.
  // All I can say is that it compiles.

  DATABLOCK_STATUS
  c_datablock_get_array_ndim(c_datablock * s, const char* section, const char * name, int * ndim)
  {
    *ndim = 0;
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;

    auto p = static_cast<DataBlock *>(s);
    vector<size_t> extents;
    // get type
    datablock_type_t dtype;
    auto status = c_datablock_get_type(s, section, name, &dtype);

  // DBT_INTND,
  // DBT_DOUBLEND,
  // DBT_COMPLEXND,

    switch (dtype){
      case DBT_INT1D:
      case DBT_DOUBLE1D:
      case DBT_COMPLEX1D:
      case DBT_STRING1D:
        *ndim=1;
        return DBS_SUCCESS;
        break;
      case DBT_INTND:
        status = p->get_array_shape<int>(section, name, extents);
        break;
      case DBT_DOUBLEND:
        status = p->get_array_shape<double>(section, name, extents);
        break;
      case DBT_COMPLEXND:
        status = p->get_array_shape<complex_t>(section, name, extents);
        break;
      default:
        status = DBS_WRONG_VALUE_TYPE;
    }

    if (status!=DBS_SUCCESS){
      *ndim = 0;
      return status;
    }

    *ndim = clamp(extents.size());
    return DBS_SUCCESS;
  }



  int c_datablock_get_array_length(c_datablock const* s,
                                   const char* section,
                                   const char* name)
  {
    if (s==nullptr || section==nullptr || name==nullptr) return -1;
    DataBlock const* p = static_cast<DataBlock const*>(s);
    return p->get_size(section, name);
  }

  const char* c_datablock_get_section_name(c_datablock const* s, int i)
  {
    if (i < 0) return nullptr;
    auto n = static_cast<size_t>(i);
    DataBlock const* p = static_cast<DataBlock const*>(s);
    if (n >= p->num_sections()) return nullptr;
    return p->section_name(n).c_str();
  }

  const char* c_datablock_get_value_name(c_datablock const* s, 
					 const char* section, int j)
  {
    if (s == nullptr) return nullptr;
    if (section == nullptr) return nullptr;
    if (j < 0) return nullptr;
    DataBlock const* p = static_cast<DataBlock const*>(s);
    const char* res = nullptr;
    try { res = p->value_name(section, j).c_str(); }
    catch (...) { }
    return res;
  }

  const char*
  c_datablock_get_value_name_by_section_index(c_datablock const* s, 
					      int i, int j)
  {
    if (s == nullptr) return nullptr;
    if (i<0 || j<0) return nullptr;
    DataBlock const* p = static_cast<DataBlock const*>(s);
    const char* res = nullptr;
    try { res = p->value_name(i, j).c_str(); }
    catch (...) { }
    return res;
  }

  DATABLOCK_STATUS destroy_c_datablock(c_datablock* s)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    DataBlock* p = static_cast<DataBlock*>(s);
    // The call to clear() is not really necessary, but to aid in
    // debugging incorrect use of the C interface (especially to help
    // detect premature calls to destroy_c_datablock), it seems
    // useful.
    p->clear();
    delete p;
    return DBS_SUCCESS;
  }

  DATABLOCK_STATUS c_datablock_get_type(c_datablock const * s,
                                        const char* section,
                                        const char* name,
                                        datablock_type_t * val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    DataBlock const* p = static_cast<DataBlock const*>(s);
    return p->get_type(section, name, *val);
  }

  DATABLOCK_STATUS
  c_datablock_get_int(c_datablock * s,
		      const char* section,
		      const char* name,
		      int* val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;

    auto p = static_cast<DataBlock *>(s);
    return p->get_val(section, name, *val);
  }

  DATABLOCK_STATUS
  c_datablock_get_int_default(c_datablock * s,
                              const char* section,
                              const char* name,
                              int def,
                              int* val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;

    auto p = static_cast<DataBlock *>(s);
    return p->get_val(section, name, def, *val);
  }


  DATABLOCK_STATUS
  c_datablock_get_bool(c_datablock * s,
          const char* section,
          const char* name,
          bool* val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;

    auto p = static_cast<DataBlock *>(s);
    return p->get_val(section, name, *val);
  }

  DATABLOCK_STATUS
  c_datablock_get_bool_default(c_datablock * s,
                              const char* section,
                              const char* name,
                              bool def,
                              bool* val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;

    auto p = static_cast<DataBlock *>(s);
    return p->get_val(section, name, def, *val);
  }



  DATABLOCK_STATUS
  c_datablock_get_double(c_datablock * s,
			 const char* section,
			 const char* name,
			 double* val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;

    auto p = static_cast<DataBlock *>(s);
    return p->get_val(section, name, *val);
  }

  DATABLOCK_STATUS
  c_datablock_get_double_default(c_datablock * s,
                                 const char* section,
                                 const char* name,
                                 double def,
                                 double* val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;

    auto p = static_cast<DataBlock *>(s);
    return p->get_val(section, name, def, *val);
  }

  DATABLOCK_STATUS
  c_datablock_get_complex(c_datablock * s,
			  const char* section,
			  const char* name,
			  double _Complex* val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;

    auto p = static_cast<DataBlock *>(s);
    complex_t z;
    auto rc = p->get_val(section, name, z);
    if (rc == DBS_SUCCESS) *val = from_complex(z);
    return rc;
  }

  DATABLOCK_STATUS
  c_datablock_get_complex_default(c_datablock * s,
                                  const char* section,
                                  const char* name,
                                  double _Complex def,
                                  double _Complex* val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;

    auto p = static_cast<DataBlock *>(s);
    complex_t default_z = from_Complex(def);
    complex_t z;
    auto rc = p->get_val(section, name, default_z, z);
    if (rc == DBS_SUCCESS) *val = from_complex(z);
    return rc;
  }

  DATABLOCK_STATUS
  c_datablock_get_string(c_datablock * s,
                         const char* section,
                         const char* name,
                         char**  val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;

    auto p = static_cast<DataBlock *>(s);
    string tmp;
    auto rc = p->get_val(section, name, tmp);
    if (rc != DBS_SUCCESS) return rc;
    *val = strdup(tmp.c_str());
    if (*val == nullptr) return DBS_MEMORY_ALLOC_FAILURE;
    return DBS_SUCCESS;
  }

  DATABLOCK_STATUS
  c_datablock_get_string_default(c_datablock * s,
                                 const char* section,
                                 const char* name,
                                 const char* def,
                                 char**  val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    /* Do we need a new enumeration value for the following? */
    if (def == nullptr) return DBS_VALUE_NULL; 
    if (val == nullptr) return DBS_VALUE_NULL;

    auto p = static_cast<DataBlock *>(s);
    string tmp;
    string default_string(def);
    auto rc = p->get_val(section, name, default_string, tmp);
    if (rc != DBS_SUCCESS) return rc;
    *val = strdup(tmp.c_str());
    if (*val == nullptr) return DBS_MEMORY_ALLOC_FAILURE;
    return DBS_SUCCESS;
  }

  DATABLOCK_STATUS
  c_datablock_get_int_array_1d(c_datablock * s,
                               const char* section,
                               const char* name,
                               int** val,
                               int* sz)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (sz == nullptr) return DBS_SIZE_NULL;

    auto p = static_cast<DataBlock *>(s);
    try {
      vector<int> const& r = p->view<vector<int>>(section, name);
      *val = static_cast<int*>(malloc(r.size() * sizeof(int)));
      if (*val ==nullptr) return DBS_MEMORY_ALLOC_FAILURE;
      std::copy(r.cbegin(), r.cend(), *val);
      *sz = r.size();
    }
    catch (DataBlock::BadDataBlockAccess const&) { return DBS_SECTION_NOT_FOUND; }
    catch (Section::BadSectionAccess const&) { return DBS_NAME_NOT_FOUND; }
    catch (Entry::BadEntry const&) { return DBS_WRONG_VALUE_TYPE; }
    catch (...) { return DBS_LOGIC_ERROR; }
    return DBS_SUCCESS;
  }

  DATABLOCK_STATUS
  c_datablock_get_double_array_1d(c_datablock * s,
                               const char* section,
                               const char* name,
                               double** val,
                               int* sz)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (sz == nullptr) return DBS_SIZE_NULL;

    auto p = static_cast<DataBlock *>(s);
    try {
      vector<double> const& r = p->view<vector<double>>(section, name);
      *val = static_cast<double*>(malloc(r.size() * sizeof(double)));
      if (*val ==nullptr) return DBS_MEMORY_ALLOC_FAILURE;
      std::copy(r.cbegin(), r.cend(), *val);
      *sz = r.size();
    }
    catch (DataBlock::BadDataBlockAccess const&) { return DBS_SECTION_NOT_FOUND; }
    catch (Section::BadSectionAccess const&) { return DBS_NAME_NOT_FOUND; }
    catch (Entry::BadEntry const&) { return DBS_WRONG_VALUE_TYPE; }
    catch (...) { return DBS_LOGIC_ERROR; }
    return DBS_SUCCESS;
  }

  DATABLOCK_STATUS
  c_datablock_get_complex_array_1d(c_datablock * s,
				   const char* section,
				   const char* name,
				   double _Complex** val,
				   int* sz)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (sz == nullptr) return DBS_SIZE_NULL;

    auto p = static_cast<DataBlock *>(s);
    try {
      vector<complex_t> const& r = p->view<vector<complex_t>>(section, name);
      *val = static_cast<double _Complex*>(malloc(r.size() * sizeof(double _Complex)));
      if (*val ==nullptr) return DBS_MEMORY_ALLOC_FAILURE;
      for (size_t i = 0, n = r.size(); i != n; ++i)
        {
          (*val)[i] = * reinterpret_cast<double _Complex const*>(&(r[i]));
        }
      *sz = r.size();
    }
    catch (DataBlock::BadDataBlockAccess const&) { return DBS_SECTION_NOT_FOUND; }
    catch (Section::BadSectionAccess const&) { return DBS_NAME_NOT_FOUND; }
    catch (Entry::BadEntry const&) { return DBS_WRONG_VALUE_TYPE; }
    catch (...) { return DBS_LOGIC_ERROR; }
    return DBS_SUCCESS;
  }

  DATABLOCK_STATUS
  c_datablock_get_str_array_1d(c_datablock * s,
                               const char* section,
                               const char* name,
                               char*** val,
                               int* sz)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (sz == nullptr) return DBS_SIZE_NULL;

    auto p = static_cast<DataBlock *>(s);
    try {
      vector<string> const& r = p->view<vector<string>>(section, name);
      *val = static_cast<char**>(malloc(r.size() * sizeof(char*)));
      *sz = r.size();
      if (*val ==nullptr) return DBS_MEMORY_ALLOC_FAILURE;
      for (int i=0; i<*sz; i++){
        (*val)[i] = strdup(r[i].c_str());
      }

      *sz = r.size();
    }
    catch (DataBlock::BadDataBlockAccess const&) { return DBS_SECTION_NOT_FOUND; }
    catch (Section::BadSectionAccess const&) { return DBS_NAME_NOT_FOUND; }
    catch (Entry::BadEntry const&) { return DBS_WRONG_VALUE_TYPE; }
    catch (...) { return DBS_LOGIC_ERROR; }
    return DBS_SUCCESS;
  }


  DATABLOCK_STATUS
  c_datablock_get_int_array_1d_preallocated(c_datablock * s,
                                            const char* section,
                                            const char* name,
                                            int* val,
                                            int* sz,
                                            int maxsize)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (sz == nullptr) return DBS_SIZE_NULL;

    auto p = static_cast<DataBlock *>(s);
    try {
      vector<int> const& r = p->view<vector<int>>(section, name);
      *sz = r.size();
      if (r.size() > static_cast<size_t>(maxsize)) return DBS_SIZE_INSUFFICIENT;
      std::copy(r.cbegin(), r.cend(), val);
    }
    catch (DataBlock::BadDataBlockAccess const&) { return DBS_SECTION_NOT_FOUND; }
    catch (Section::BadSectionAccess const&) { return DBS_NAME_NOT_FOUND; }
    catch (Entry::BadEntry const&) { return DBS_WRONG_VALUE_TYPE; }
    catch (...) { return DBS_LOGIC_ERROR; }
    // If we are asked to clear out the remainder of the input buffer,
    // the following line should be used.
    //    std::fill(val + *sz, val+maxsize, 0);
    return DBS_SUCCESS;
  }


  DATABLOCK_STATUS
  c_datablock_get_double_array_1d_preallocated(c_datablock * s,
                                            const char* section,
                                            const char* name,
                                            double* val,
                                            int* sz,
                                            int maxsize)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (sz == nullptr) return DBS_SIZE_NULL;

    auto p = static_cast<DataBlock *>(s);
    try {
      vector<double> const& r = p->view<vector<double>>(section, name);
      *sz = r.size();
      if (r.size() > static_cast<size_t>(maxsize)) return DBS_SIZE_INSUFFICIENT;
      std::copy(r.cbegin(), r.cend(), val);
    }
    catch (DataBlock::BadDataBlockAccess const&) { return DBS_SECTION_NOT_FOUND; }
    catch (Section::BadSectionAccess const&) { return DBS_NAME_NOT_FOUND; }
    catch (Entry::BadEntry const&) { return DBS_WRONG_VALUE_TYPE; }
    catch (...) { return DBS_LOGIC_ERROR; }
    // If we are asked to clear out the remainder of the input buffer,
    // the following line should be used.
    //    std::fill(val + *sz, val+maxsize, 0);
    return DBS_SUCCESS;
  }


  DATABLOCK_STATUS
  c_datablock_get_complex_array_1d_preallocated(c_datablock * s,
						const char* section,
						const char* name,
						double _Complex * val,
						int* sz,
						int maxsize)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (sz == nullptr) return DBS_SIZE_NULL;

    auto p = static_cast<DataBlock *>(s);
    try{
      vector<complex_t> const& r = p->view<vector<complex_t>>(section, name);
      *sz = r.size();
      if (r.size() > static_cast<size_t>(maxsize)) return DBS_SIZE_INSUFFICIENT;
      //std::copy(r.cbegin(), r.cend(), val);
      for (size_t i = 0, n = r.size(); i != n; ++i)
        {
          val[i] = from_complex(r[i]);
        }
    }
    catch (DataBlock::BadDataBlockAccess const&) { return DBS_SECTION_NOT_FOUND; }
    catch (Section::BadSectionAccess const&) { return DBS_NAME_NOT_FOUND; }
    catch (Entry::BadEntry const&) { return DBS_WRONG_VALUE_TYPE; }
    catch (...) { return DBS_LOGIC_ERROR; }


    // If we are asked to clear out the remainder of the input buffer,
    // the following line should be used.
    //    std::fill(val + *sz, val+maxsize, 0);
    return DBS_SUCCESS;
  }


  DATABLOCK_STATUS
  c_datablock_get_str_array_1d_preallocated(c_datablock * s,
                               const char* section,
                               const char* name,
                               char** val,
                               int* sz)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (sz == nullptr) return DBS_SIZE_NULL;

    auto p = static_cast<DataBlock *>(s);
    try {
      vector<string> const& r = p->view<vector<string>>(section, name);
      *sz = r.size();
      for (int i=0; i<*sz; i++){
        val[i] = strdup(r[i].c_str());
      }
    }
    catch (DataBlock::BadDataBlockAccess const&) { return DBS_SECTION_NOT_FOUND; }
    catch (Section::BadSectionAccess const&) { return DBS_NAME_NOT_FOUND; }
    catch (Entry::BadEntry const&) { return DBS_WRONG_VALUE_TYPE; }
    catch (...) { return DBS_LOGIC_ERROR; }
    return DBS_SUCCESS;
  }

  DATABLOCK_STATUS
  c_datablock_put_int(c_datablock* s,
          const char* section,
          const char* name,
          int val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;

    auto p = static_cast<DataBlock*>(s);
    return p->put_val(section, name, val);
  }

  DATABLOCK_STATUS
  c_datablock_put_bool(c_datablock* s,
          const char* section,
          const char* name,
          bool val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;

    auto p = static_cast<DataBlock*>(s);
    return p->put_val(section, name, val);
  }

  DATABLOCK_STATUS
  c_datablock_put_double(c_datablock* s,
			 const char* section,
			 const char* name,
			 double val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;

    auto p = static_cast<DataBlock*>(s);
    return p->put_val(section, name, val);
  }

  DATABLOCK_STATUS
  c_datablock_put_complex(c_datablock* s,
			  const char* section,
			  const char* name,
			  double _Complex val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;

    auto p = static_cast<DataBlock*>(s);
    complex_t z = from_Complex(val);
    return p->put_val(section, name, z);
  }

  DATABLOCK_STATUS
  c_datablock_put_string(c_datablock* s,
			 const char* section,
			 const char* name,
			 const char* val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == NULL) return DBS_VALUE_NULL;

    auto p = static_cast<DataBlock*>(s);
    return p->put_val(section, name, string(val));
  }

  DATABLOCK_STATUS
  c_datablock_put_int_array_1d(c_datablock* s,
                               const char* section,
                               const char* name,
                               int const*  val,
                               int sz)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == NULL) return DBS_VALUE_NULL;
    if (sz < 1) return DBS_SIZE_NONPOSITIVE;

    auto p = static_cast<DataBlock*>(s);
    return p->put_val(section, name, vector<int>(val, val+sz));
  }

  DATABLOCK_STATUS
  c_datablock_put_double_array_1d(c_datablock* s,
                               const char* section,
                               const char* name,
                               double const*  val,
                               int sz)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == NULL) return DBS_VALUE_NULL;
    if (sz < 1) return DBS_SIZE_NONPOSITIVE;

    auto p = static_cast<DataBlock*>(s);
    return p->put_val(section, name, vector<double>(val, val+sz));
  }

  DATABLOCK_STATUS
  c_datablock_put_complex_array_1d(c_datablock* s,
                               const char* section,
                               const char* name,
                               double _Complex const*  val,
                               int sz)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == NULL) return DBS_VALUE_NULL;
    if (sz < 1) return DBS_SIZE_NONPOSITIVE;

    auto p = static_cast<DataBlock*>(s);
    std::vector<complex_t> zs = from_Complex(val, sz);
    return p->put_val(section, name, zs);
  }


    DATABLOCK_STATUS
    c_datablock_put_str_array_1d(c_datablock* s,
                               const char* section,
                               const char* name,
                               const char * const* val,
                               int sz)
    {
        if (s == nullptr) return DBS_DATABLOCK_NULL;
        if (section == nullptr) return DBS_SECTION_NULL;
        if (name == nullptr) return DBS_NAME_NULL;
        if (val == NULL) return DBS_VALUE_NULL;
        if (sz < 1) return DBS_SIZE_NONPOSITIVE;

        auto p = static_cast<DataBlock*>(s);

        vector<string> v(val, val + sz);
        return p->put_val(section, name, v);
    }

  DATABLOCK_STATUS
  c_datablock_replace_int(c_datablock* s,
        const char* section,
        const char* name,
        int val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;

    auto p = static_cast<DataBlock*>(s);
    return p->replace_val(section, name, val);
  }

  DATABLOCK_STATUS
  c_datablock_replace_bool(c_datablock* s,
        const char* section,
        const char* name,
        bool val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;

    auto p = static_cast<DataBlock*>(s);
    return p->replace_val(section, name, val);
  }


  DATABLOCK_STATUS
  c_datablock_replace_double(c_datablock* s,
			     const char* section,
			     const char* name,
			     double val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;

    auto p = static_cast<DataBlock*>(s);
    return p->replace_val(section, name, val);
  }

  DATABLOCK_STATUS
  c_datablock_replace_complex(c_datablock* s,
			     const char* section,
			     const char* name,
			     double _Complex val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;

    auto p = static_cast<DataBlock*>(s);
    complex_t z = from_Complex(val);
    return p->replace_val(section, name, z);
  }

  DATABLOCK_STATUS
  c_datablock_replace_string(c_datablock* s,
			     const char* section,
			     const char* name,
			     const char* val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == NULL) return DBS_VALUE_NULL;

    auto p = static_cast<DataBlock*>(s);
    return p->replace_val(section, name, string(val));
  }

  DATABLOCK_STATUS
  c_datablock_replace_int_array_1d(c_datablock* s,
                                   const char* section,
                                   const char* name,
                                   int const* val,
                                   int sz)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (sz  < 1) return DBS_SIZE_NONPOSITIVE;

    auto p = static_cast<DataBlock*>(s);
    return p->replace_val(section, name, vector<int>(val, val+sz));
  }

  DATABLOCK_STATUS
  c_datablock_replace_double_array_1d(c_datablock* s,
                                   const char* section,
                                   const char* name,
                                   double const* val,
                                   int sz)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (sz  < 1) return DBS_SIZE_NONPOSITIVE;

    auto p = static_cast<DataBlock*>(s);
    return p->replace_val(section, name, vector<double>(val, val+sz));
  }

  DATABLOCK_STATUS
  c_datablock_replace_complex_array_1d(c_datablock* s,
				       const char* section,
				       const char* name,
				       double _Complex const* val,
				       int sz)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (sz  < 1) return DBS_SIZE_NONPOSITIVE;

    auto p = static_cast<DataBlock*>(s);
    vector<complex_t> zs = from_Complex(val, sz);
    return p->replace_val(section, name, zs);
  }

  DATABLOCK_STATUS
  c_datablock_put_int_array(c_datablock* s,
                            const char* section,
                            const char* name,
                            int const* val,
                            int ndims,
                            int const* extents)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (ndims  < 1) return DBS_NDIM_NONPOSITIVE;
    if (extents == nullptr) return DBS_EXTENTS_NULL;

    auto p = static_cast<DataBlock*>(s);
    ndarray<int> tmp(val, ndims, extents);
    return p->put_val(section, name, tmp);
  }

    DATABLOCK_STATUS
  c_datablock_replace_int_array(c_datablock* s,
                            const char* section,
                            const char* name,
                            int const* val,
                            int ndims,
                            int const* extents)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (ndims  < 1) return DBS_NDIM_NONPOSITIVE;
    if (extents == nullptr) return DBS_EXTENTS_NULL;

    auto p = static_cast<DataBlock*>(s);
    ndarray<int> tmp(val, ndims, extents);
    return p->replace_val(section, name, tmp);
  }


  DATABLOCK_STATUS
  c_datablock_get_int_array_shape(c_datablock* s,
                                  const char* section,
                                  const char* name,
                                  int ndims,
                                  int* extents)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (ndims  < 1) return DBS_NDIM_NONPOSITIVE;
    if (extents == nullptr) return DBS_EXTENTS_NULL;

    auto p = static_cast<DataBlock*>(s);
    vector<size_t> local_extents;
    auto status = p->get_array_shape<int>(section, name, local_extents);
    if (status != DBS_SUCCESS) return status;
    if (clamp(local_extents.size()) != ndims) return DBS_NDIM_MISMATCH;
    //for (size_t i = 0; i != ndims; ++i) extents[i] = local_extents[i];
    for (auto ext : local_extents) *extents++ = ext;
    return DBS_SUCCESS;
  }

  DATABLOCK_STATUS
  c_datablock_get_int_array(c_datablock* s,
                            const char* section,
                            const char* name,
                            int* val,
                            int ndims,
                            int const* extents)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (ndims  < 1) return DBS_NDIM_NONPOSITIVE;
    if (extents == nullptr) return DBS_EXTENTS_NULL;

    auto p = static_cast<DataBlock *>(s);
    try {
      ndarray<int> const& r = p->view<ndarray<int>>(section, name);
      if (clamp(r.ndims()) != ndims) return DBS_NDIM_MISMATCH;
      for (size_t i = 0, sz = ndims; i != sz; ++i)
        if (clamp(r.extents()[i]) != extents[i])
          return DBS_EXTENTS_MISMATCH;
      std::copy(r.begin(), r.end(), val);
    }
    catch (DataBlock::BadDataBlockAccess const&) { return DBS_SECTION_NOT_FOUND; }
    catch (Section::BadSectionAccess const&) { return DBS_NAME_NOT_FOUND; }
    catch (Entry::BadEntry const&) { return DBS_WRONG_VALUE_TYPE; }
    catch (...) { return DBS_LOGIC_ERROR; }
    return DBS_SUCCESS;
  }

  DATABLOCK_STATUS
  c_datablock_put_double_array(c_datablock* s,
                               const char* section,
                               const char* name,
                               double const* val,
                               int ndims,
                               int const* extents)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (ndims  < 1) return DBS_NDIM_NONPOSITIVE;
    if (extents == nullptr) return DBS_EXTENTS_NULL;

    auto p = static_cast<DataBlock*>(s);
    ndarray<double> tmp(val, ndims, extents);
    return p->put_val(section, name, tmp);
  }


  DATABLOCK_STATUS
  c_datablock_replace_double_array(c_datablock* s,
                               const char* section,
                               const char* name,
                               double const* val,
                               int ndims,
                               int const* extents)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (ndims  < 1) return DBS_NDIM_NONPOSITIVE;
    if (extents == nullptr) return DBS_EXTENTS_NULL;

    auto p = static_cast<DataBlock*>(s);
    ndarray<double> tmp(val, ndims, extents);
    return p->replace_val(section, name, tmp);
  }



  DATABLOCK_STATUS
  c_datablock_get_double_array_shape(c_datablock* s,
                                     const char* section,
                                     const char* name,
                                     int ndims,
                                     int* extents)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (ndims  < 1) return DBS_NDIM_NONPOSITIVE;
    if (extents == nullptr) return DBS_EXTENTS_NULL;

    auto p = static_cast<DataBlock*>(s);
    vector<size_t> local_extents;
    auto status = p->get_array_shape<double>(section, name, local_extents);
    if (status != DBS_SUCCESS) return status;
    if (clamp(local_extents.size()) != ndims) return DBS_NDIM_MISMATCH;
    //for (size_t i = 0; i != ndims; ++i) extents[i] = local_extents[i];
    for (auto ext : local_extents) *extents++ = ext;
    return DBS_SUCCESS;
  }

  DATABLOCK_STATUS
  c_datablock_get_double_array(c_datablock* s,
                               const char* section,
                               const char* name,
                               double* val,
                               int ndims,
                               int const* extents)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (ndims  < 1) return DBS_NDIM_NONPOSITIVE;
    if (extents == nullptr) return DBS_EXTENTS_NULL;

    auto p = static_cast<DataBlock *>(s);
    try {
      ndarray<double> const& r = p->view<ndarray<double>>(section, name);
      if (clamp(r.ndims()) != ndims) return DBS_NDIM_MISMATCH;
      for (size_t i = 0, sz = ndims; i != sz; ++i){
        if (clamp(r.extents()[i]) != extents[i]){
          return DBS_EXTENTS_MISMATCH;
        }
      }
      std::copy(r.begin(), r.end(), val);
     }
    catch (DataBlock::BadDataBlockAccess const&) { return DBS_SECTION_NOT_FOUND; }
    catch (Section::BadSectionAccess const&) { return DBS_NAME_NOT_FOUND; }
    catch (Entry::BadEntry const&) { return DBS_WRONG_VALUE_TYPE; }
    catch (...) { return DBS_LOGIC_ERROR; }
    return DBS_SUCCESS;
  }

  DATABLOCK_STATUS
  c_datablock_put_complex_array(c_datablock* s,
                                const char* section,
                                const char* name,
                                double _Complex const* val,
                                int ndims,
                                int const* extents)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (ndims  < 1) return DBS_NDIM_NONPOSITIVE;
    if (extents == nullptr) return DBS_EXTENTS_NULL;

    auto p = static_cast<DataBlock*>(s);
    // Complex values require translation, so this is handled
    // differently from int and double...
    size_t num_elements =
      std::accumulate(extents, extents+ndims, 1, std::multiplies<int>());

    //vector<complex_t> z(val, val + num_elements);
    vector<complex_t> z = from_Complex(val, num_elements);
    vector<size_t> local_extents(extents, extents+ndims);
    ndarray<complex_t> tmp(z, local_extents);
    return p->put_val(section, name, tmp);
  }

  DATABLOCK_STATUS
  c_datablock_get_complex_array_shape(c_datablock* s,
                                      const char* section,
                                      const char* name,
                                      int ndims,
                                      int* extents)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (ndims  < 1) return DBS_NDIM_NONPOSITIVE;
    if (extents == nullptr) return DBS_EXTENTS_NULL;

    auto p = static_cast<DataBlock*>(s);
    vector<size_t> local_extents;
    auto status = p->get_array_shape<complex_t>(section, name, local_extents);
    if (status != DBS_SUCCESS) return status;
    if (clamp(local_extents.size()) != ndims) return DBS_NDIM_MISMATCH;
    //for (size_t i = 0; i != ndims; ++i) extents[i] = local_extents[i];
    for (auto ext : local_extents) *extents++ = ext;
    return DBS_SUCCESS;
  }

  DATABLOCK_STATUS
  c_datablock_get_complex_array(c_datablock* s,
                                const char* section,
                                const char* name,
                                double _Complex* val,
                                int ndims,
                                int const* extents)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (ndims  < 1) return DBS_NDIM_NONPOSITIVE;
    if (extents == nullptr) return DBS_EXTENTS_NULL;

    auto p = static_cast<DataBlock *>(s);
    try {
      ndarray<complex_t> const& r = p->view<ndarray<complex_t>>(section, name);
      if (clamp(r.ndims()) != ndims) return DBS_NDIM_MISMATCH;
      for (size_t i = 0, sz = ndims; i != sz; ++i)
        if (clamp(r.extents()[i]) != extents[i])
          return DBS_EXTENTS_MISMATCH;
      // We rely on the layout of std::complex<double> and double
      // _Complex matching. Note that &*r.begin() returns the address of
      // the first location of the stored complex numbers.
      memcpy(val, &*r.begin(), r.size()*sizeof(complex_t));
     }
    catch (DataBlock::BadDataBlockAccess const&) { return DBS_SECTION_NOT_FOUND; }
    catch (Section::BadSectionAccess const&) { return DBS_NAME_NOT_FOUND; }
    catch (Entry::BadEntry const&) { return DBS_WRONG_VALUE_TYPE; }
    catch (...) { return DBS_LOGIC_ERROR; }
    return DBS_SUCCESS;
  }

DATABLOCK_STATUS  c_datablock_put_double_grid(
  c_datablock* s,
  const char * section, 
  const char * name_x, int n_x, double * x,  
  const char * name_y, int n_y, double * y, 
  const char * name_z, double ** z)
{
    DATABLOCK_STATUS status=DBS_SUCCESS;

    // const int ndim = 2;
    // int dims[ndim] = {n_x, n_y};

    status = c_datablock_put_double_array_1d(s, section, name_x, x, n_x);
    if (status) {return status;}
    status = c_datablock_put_double_array_1d(s, section, name_y, y, n_y);
    if (status) {return status;}
    // status |= c_datablock_put_double_array(s, section, name_z, z, ndim, dims);
    int n_z = n_x * n_y;
    double * z_flat = (double*)malloc(sizeof(double)*n_z);
    int p=0;
    for (int i=0; i<n_x; i++){
      for (int j=0; j<n_y; j++){
        z_flat[p++] = z[i][j];
      }
    }
    int ndim=2;
    int extents[2] = {n_x, n_y};
    status = c_datablock_put_double_array(s, section, name_z, z_flat, ndim, extents);
    free(z_flat);
    if (status) {return status;}

    // We could rely on n_x and n_y to record in the block what ordering the array has.
    // But that would break down if n_x == n_y
    char sentinel_key[512];
    char sentinel_value[512];

    snprintf(sentinel_key, 512, "_cosmosis_order_%s",name_z);
    snprintf(sentinel_value, 512, "%s_cosmosis_order_%s",name_x, name_y);

    status = c_datablock_put_string(s, section, sentinel_key, sentinel_value);
    return status;
}


DATABLOCK_STATUS  c_datablock_replace_double_grid(
  c_datablock* s,
  const char * section, 
  const char * name_x, int n_x, double * x,  
  const char * name_y, int n_y, double * y, 
  const char * name_z, double ** z)
{
    DATABLOCK_STATUS status=DBS_SUCCESS;

    // const int ndim = 2;
    // int dims[ndim] = {n_x, n_y};

    status = c_datablock_replace_double_array_1d(s, section, name_x, x, n_x);
    if (status) {return status;}
    status = c_datablock_replace_double_array_1d(s, section, name_y, y, n_y);
    if (status) {return status;}
    // status |= c_datablock_put_double_array(s, section, name_z, z, ndim, dims);
    int n_z = n_x * n_y;
    double * z_flat = (double*)malloc(sizeof(double)*n_z);
    int p=0;
    for (int i=0; i<n_x; i++){
      for (int j=0; j<n_y; j++){
        z_flat[p++] = z[i][j];
      }
    }

    int ndim=2;
    int extents[2] = {n_y, n_x};
    status = c_datablock_replace_double_array(s, section, name_z, z_flat, ndim, extents);
    free(z_flat);
    if (status) {return status;}

    // We could rely on n_x and n_y to record in the block what ordering the array has.
    // But that would break down if n_x == n_y
    char sentinel_key[512];
    char sentinel_value[512];

    snprintf(sentinel_key, 512, "_cosmosis_order_%s",name_z);
    snprintf(sentinel_value, 512, "%s_cosmosis_order_%s",name_x, name_y);

    status = c_datablock_replace_string(s, section, sentinel_key, sentinel_value);
    return status;
}



double ** allocate_2d_double(int nx, int ny){
  double ** z = (double**)malloc(nx*sizeof(double*));
  for (int i=0; i<nx; i++){
    z[i] = (double*)malloc(sizeof(double)*ny);
  }
  return z;
}

void deallocate_2d_double(double *** z, int nx){
  for (int i=0; i<nx; i++){
    free((*z)[i]);
  }
  free(*z);
  *z = NULL;
}

DATABLOCK_STATUS  c_datablock_get_double_grid(
  c_datablock* s,
  const char * section, 
  const char * name_x, int *n_x, double ** x,  
  const char * name_y, int *n_y, double ** y, 
  const char * name_z, double *** z)
{
    DATABLOCK_STATUS status;
    *x = NULL;
    *y = NULL;
    *z = NULL;
    int nx, ny;

    status = c_datablock_get_double_array_1d(s, section, name_x, x, &nx);
    if (status) {return status;}
    status = c_datablock_get_double_array_1d(s, section, name_y, y, &ny);
    if (status) {free(*x); *x=NULL; return status;}

    double * z_flat = (double*)malloc(nx*ny*sizeof(double));

    //Now we need to check if the ordering requested here is the same
    //as the saved ordering.  If not we need to transpose.
    char sentinel_key[512];
    char * sentinel_value;
    char sentinel_test[512];

    double ** z_2d = allocate_2d_double(nx, ny);

    snprintf(sentinel_key, 512, "_cosmosis_order_%s",name_z);
    status = c_datablock_get_string(s,section, sentinel_key, &sentinel_value);
    if (status) {free(*x); free(*y); free(z_flat); *x=NULL; *y=NULL; deallocate_2d_double(&z_2d, nx); return status;}
  
    snprintf(sentinel_test, 512, "%s_cosmosis_order_%s",name_x, name_y);
    if (0==strncmp(sentinel_test, sentinel_value, 512)){
      // This indicates that the requested ordering is the same as the stored one.
      // So we do not need to do any flipping.
      int ndim=2;
      int extents[2] = {nx, ny};

      status = c_datablock_get_double_array(s, section, name_z, z_flat, ndim, extents);
      if (status) {free(*x); free(*y); *x=NULL; *y=NULL; deallocate_2d_double(&z_2d, nx); return status;}

      for (int i=0; i<nx; i++){
        for (int j=0; j<ny; j++){
          z_2d[i][j] = z_flat[i*ny+j];
        }
      }
    }
    else{
      snprintf(sentinel_test, 512, "%s_cosmosis_order_%s",name_y, name_x);
      int ndim=2;
      int extents[2] = {ny, nx};

      status = c_datablock_get_double_array(s, section, name_z, z_flat, ndim, extents);
      if (status) {free(*x); free(*y); *x=NULL; *y=NULL; deallocate_2d_double(&z_2d, nx); return status;}

      if (0==strncmp(sentinel_test, sentinel_value, 512)){
        for (int i=0; i<nx; i++){
          for (int j=0; j<ny; j++){
            z_2d[i][j] = z_flat[j*nx+i];
          }
        }
      }
      else{
        // no match - something wrong. 
        status = DBS_WRONG_VALUE_TYPE;
        free(*x); 
        free(*y); 
        free(z_flat); 
        *x=NULL; 
        *y=NULL; 
        deallocate_2d_double(&z_2d, nx);
        free(sentinel_value);     
        return status;
      }
    }
  free(sentinel_value);
  free(z_flat);

  *n_x = nx;
  *n_y = ny;
  *z = z_2d;
  return status;
}


DATABLOCK_STATUS
c_datablock_report_failures(c_datablock* s)
{
  if (s == nullptr) return DBS_DATABLOCK_NULL;
  auto p = static_cast<DataBlock*>(s);
  cerr << "--- Error log --- " << endl;
  p->report_failures(cerr);
  cerr << "--- End log --- " << endl;
  return DBS_SUCCESS;
}



DATABLOCK_STATUS
c_datablock_print_log(c_datablock* s)
{
  if (s == nullptr) return DBS_DATABLOCK_NULL;
  auto p = static_cast<DataBlock*>(s);
  cout << "--- Access log --- " << endl;
  p->print_log();
  cout << "--- End log --- " << endl;
  return DBS_SUCCESS;
}


DATABLOCK_STATUS
c_datablock_log_access(c_datablock* s, 
                       const char * log_type,
                       const char* section,
                       const char* name)
{

  if (s == nullptr) return DBS_DATABLOCK_NULL;
  string t = string(""); // Dummy type since not posible in C
  auto p = static_cast<DataBlock*>(s);
  p->log_access(log_type, section, name, typeid(t));
  return DBS_SUCCESS;
}

int c_datablock_get_log_count(c_datablock *s)
{
    if (s == nullptr) return -1;
    auto p = static_cast<DataBlock*>(s);
    return p->get_log_count();


}

DATABLOCK_STATUS
c_datablock_get_log_entry(c_datablock* s,
                          int i,
                          int smax,
                          char *log_type,
                          char *section,
                          char *name,
                          char *dtype
  )
{
  
  if (s == nullptr) return DBS_DATABLOCK_NULL;
  auto p = static_cast<DataBlock*>(s);
  std::string log_type_string, section_string, name_string, dtype_string;
  DATABLOCK_STATUS status = p->get_log_entry(i, 
    log_type_string, section_string, name_string, dtype_string);

  if (status) return status;

  strncpy(log_type, log_type_string.c_str(), smax);
  strncpy(section, section_string.c_str(), smax);
  strncpy(name, name_string.c_str(), smax);
  strncpy(dtype, dtype_string.c_str(), smax);

  return DBS_SUCCESS;
}


DATABLOCK_STATUS
c_datablock_get_metadata(c_datablock* s, 
                       const char* section,
                       const char* name,
                       const char* key,
                       char** val
                       )
{
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (key == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;

    auto p = static_cast<DataBlock *>(s);
    string tmp;
    auto rc = p->get_metadata(section, name, key, tmp);
    if (rc != DBS_SUCCESS) return rc;
    *val = strdup(tmp.c_str());
    if (*val == nullptr) return DBS_MEMORY_ALLOC_FAILURE;
    return DBS_SUCCESS;
}

DATABLOCK_STATUS
c_datablock_put_metadata(c_datablock* s,
     const char* section,
     const char* name,
     const char* key,
     const char* val)
{
  if (s == nullptr) return DBS_DATABLOCK_NULL;
  if (section == nullptr) return DBS_SECTION_NULL;
  if (name == nullptr) return DBS_NAME_NULL;
  if (key == nullptr) return DBS_NAME_NULL;
  if (val == NULL) return DBS_VALUE_NULL;

  auto p = static_cast<DataBlock*>(s);
  return p->put_metadata(section, name, key, val);
}


DATABLOCK_STATUS
c_datablock_replace_metadata(c_datablock* s,
     const char* section,
     const char* name,
     const char* key,
     const char* val)
{
  if (s == nullptr) return DBS_DATABLOCK_NULL;
  if (section == nullptr) return DBS_SECTION_NULL;
  if (name == nullptr) return DBS_NAME_NULL;
  if (key == nullptr) return DBS_NAME_NULL;
  if (val == NULL) return DBS_VALUE_NULL;

  auto p = static_cast<DataBlock*>(s);
  return p->replace_metadata(section, name, key, val);
}


DATABLOCK_STATUS
c_datablock_serialize(c_datablock const* s, const char* path)
{
  if (s == nullptr) return DBS_DATABLOCK_NULL;
  if (path == nullptr) return DBS_NAME_NULL;
  auto p = static_cast<DataBlock const*>(s);
  return p->serialize(std::string(path)) ? DBS_SUCCESS : DBS_LOGIC_ERROR;
}

DATABLOCK_STATUS
c_datablock_deserialize(c_datablock* s, const char* path)
{
  if (s == nullptr) return DBS_DATABLOCK_NULL;
  if (path == nullptr) return DBS_NAME_NULL;
  auto p = static_cast<DataBlock*>(s);
  return p->deserialize(std::string(path)) ? DBS_SUCCESS : DBS_LOGIC_ERROR;
}


} // extern "C"
