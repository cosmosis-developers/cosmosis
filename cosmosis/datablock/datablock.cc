#include "datablock.hh"
#include "clamp.hh"
#include <iostream>
#include <fstream>
#include "cxxabi.h"
#include <cstdint>
#include <cstring>
#include <typeindex>
using namespace std;

bool cosmosis::DataBlock::has_val(string section,
                                              string name) const
{
  downcase(section); downcase(name);
  auto isec = sections_.find(section);
  if (isec == sections_.end()) return false;
  return isec->second.has_val(name) ? true : false;
}

int cosmosis::DataBlock::get_size(string section,
                                  string name) const
{
  downcase(section); downcase(name);
  auto isec = sections_.find(section);
  if (isec == sections_.end()) return -1;
  return isec->second.get_size(name);
}

DATABLOCK_STATUS cosmosis::DataBlock::get_type(string section,
                                              string name, datablock_type_t &t) const
{
  downcase(section); downcase(name);
  auto isec = sections_.find(section);
  if (isec == sections_.end()) return DBS_SECTION_NOT_FOUND;
  return isec->second.get_type(name,t);
}


bool cosmosis::DataBlock::has_section(string name) const
{
  downcase(name);
  return sections_.find(name) != sections_.end();
}

int cosmosis::DataBlock::num_values(string section) const
{
  downcase(section);
  auto isec = sections_.find(section);
  if (isec == sections_.end()) return -1;
  return clamp(isec->second.number_values());
}

std::size_t cosmosis::DataBlock::num_sections() const
{
  return sections_.size();
}

std::string const& cosmosis::DataBlock::section_name(std::size_t i) const
{
  if (i >= num_sections()) throw BadDataBlockAccess();
  auto isec = sections_.begin();
  std::advance(isec, i);
  return isec->first;

}


std::string const& cosmosis::DataBlock::value_name(int i, int j) const
{
  std::string section = section_name(i);
  return value_name(section,j);
}


std::string const& cosmosis::DataBlock::value_name(std::string section, int j) const
{
  downcase(section);
  auto isec = sections_.find(section);
  if (isec == sections_.end()) throw BadDataBlockAccess();
  return isec->second.value_name(j);
}


void cosmosis::DataBlock::print_log()
{
  for (auto L=access_log_.begin(); L!=access_log_.end(); ++L){
    auto l = *L;
    auto access_type = std::get<0>(l);
    auto section = std::get<1>(l);
    auto name = std::get<2>(l);
    bool new_module = access_type == std::string(BLOCK_LOG_START_MODULE);
    if (new_module) std::cout << std::endl << std::endl;
      std::cout << access_type << "    " << section << "    " << name << std::endl;
    if (new_module) std::cout << std::endl;
  }

}

DATABLOCK_STATUS 
cosmosis::DataBlock::copy_section(std::string source, std::string dest)
{
  downcase(source); downcase(dest);
  if (!has_section(source)) return DBS_SECTION_NOT_FOUND;
  if (has_section(dest)) return DBS_NAME_ALREADY_EXISTS;  //slight abuse
  auto& source_section = sections_[source];
  auto& dest_section = source_section;
  sections_[dest] = dest_section;
  log_access(BLOCK_LOG_COPY, source, dest, typeid(source));
  return DBS_SUCCESS;
}


void cosmosis::DataBlock::clear()
{
  std::string t = std::string("");
  log_access(BLOCK_LOG_CLEAR, "", "", typeid(t));
  sections_.clear();
}

DATABLOCK_STATUS 
cosmosis::DataBlock::delete_section(std::string section)
{
  downcase(section);
  auto isec = sections_.find(section);
  if (isec == sections_.end()) return DBS_SECTION_NOT_FOUND;
  sections_.erase(isec);
  std::string t = std::string("");
  log_access(BLOCK_LOG_DELETE, section, "", typeid(t));

  return DBS_SUCCESS;
}

void cosmosis::DataBlock::log_access(const std::string& log_type, 
  const std::string& section, const std::string &name, const std::type_info& type)
{
  auto entry = log_entry(log_type, section, name, type);
  access_log_.push_back(entry);
}

int cosmosis::DataBlock::get_log_count()
{
  return access_log_.size();
}


DATABLOCK_STATUS
cosmosis::DataBlock::get_log_entry(int i, 
  std::string& log_type, 
  std::string& section, 
  std::string& name, 
  std::string& type)
{
  if (i<0) return DBS_SIZE_INSUFFICIENT;
  unsigned int j = (unsigned int) i;
  if (j>=access_log_.size()) return DBS_SIZE_INSUFFICIENT;
  const log_entry entry = access_log_[j];
  log_type = std::get<0>(entry);
  section = std::get<1>(entry);
  name = std::get<2>(entry);
  std::type_index info(std::get<3>(entry));
  // __cxa_demangle requires that the buffer is allocated with malloc,
  // as it will sometimes reallocate the buffer to fit the string.
  size_t len = 128;
  char * type_name = (char *) malloc(len);
  int status;
  char * type_name2 = abi::__cxa_demangle(info.name(), type_name, &len, &status);

  if (status){
    type = info.name();
  }
  else{
    type = type_name2;
  }
  free(type_name2);

  return DBS_SUCCESS;
}

void cosmosis::DataBlock::report_failures(std::ostream &output)
{
   for (auto L=access_log_.begin(); L!=access_log_.end(); ++L){
      auto l = *L;
      auto access_type = std::get<0>(l);
      auto section = std::get<1>(l);
      auto name = std::get<2>(l);
      if(access_type==BLOCK_LOG_READ_FAIL){
        output << "Failed to read " << name << " from " << section << std::endl;
      }
      if(access_type==BLOCK_LOG_WRITE_FAIL){
        output << "Failed to write " << name << " into " << section << std::endl;
      }
      if(access_type==BLOCK_LOG_REPLACE_FAIL){
        output << "Failed to replace " << name << " into " << section << std::endl;
      }
    }
  }


DATABLOCK_STATUS
cosmosis::DataBlock::put_metadata(std::string section,
                             std::string name,
                             std::string key,
                             std::string value)
{
    downcase(section); 
    downcase(name);

    // The thing which we are putting the metadata for must exist
    if (!has_val(section, name)) return DBS_NAME_NOT_FOUND; 

    std::string metadata_key = std::string(COSMOSIS_METADATA_PREFIX) + name + ":" + key + ":";
    return put_val(section, metadata_key, value);

}

DATABLOCK_STATUS
cosmosis::DataBlock::replace_metadata(std::string section,
                             std::string name,
                             std::string key,
                             std::string value)
{
    downcase(section); 
    downcase(name);

    // The thing which we are putting the metadata for must exist
    if (!has_val(section, name)) return DBS_NAME_NOT_FOUND; 

    std::string metadata_key = std::string(COSMOSIS_METADATA_PREFIX) + name + ":" + key + ":";
    return replace_val(section, metadata_key, value);

}

DATABLOCK_STATUS
cosmosis::DataBlock::get_metadata(std::string section,
                             std::string name,
                             std::string key,
                             std::string &value)
{
    downcase(section); 
    downcase(name);

    // The thing which we are putting the metadata for must exist
    if (!has_val(section, name)) return DBS_NAME_NOT_FOUND; 

    std::string metadata_key = std::string(COSMOSIS_METADATA_PREFIX) + name + ":" + key + ":";
    return get_val(section, metadata_key, value);

}

// ---------------------------------------------------------------------------
// Binary serialization helpers (anonymous namespace, internal linkage only)
// ---------------------------------------------------------------------------
namespace {

// Format constants
static const uint64_t DATABLOCK_SERIAL_MAGIC   = 0x434F534D4F534253ULL; // "COSMOSBS"
static const uint32_t DATABLOCK_SERIAL_VERSION = 1;

// Simple append-only binary writer backed by an external buffer.
class BinaryWriter {
public:
    explicit BinaryWriter(std::vector<uint8_t>& buf) : buf_(buf) {}

    // Write a single trivially-copyable value.
    template <typename T>
    void write(const T& val) {
        size_t off = buf_.size();
        buf_.resize(off + sizeof(T));
        std::memcpy(buf_.data() + off, &val, sizeof(T));
    }

    // Write raw bytes.
    void write_bytes(const void* data, size_t n) {
        size_t off = buf_.size();
        buf_.resize(off + n);
        std::memcpy(buf_.data() + off, data, n);
    }

    // Write a string as [uint32 length][bytes].
    void write_string(const std::string& s) {
        write(static_cast<uint32_t>(s.size()));
        write_bytes(s.data(), s.size());
    }

    // Write a vector of trivially-copyable elements as [uint32 count][raw bytes].
    template <typename T>
    void write_trivial_vector(const std::vector<T>& v) {
        write(static_cast<uint32_t>(v.size()));
        if (!v.empty()) write_bytes(v.data(), v.size() * sizeof(T));
    }

private:
    std::vector<uint8_t>& buf_;
};

// Simple read-only cursor over a byte buffer.
class BinaryReader {
public:
    BinaryReader(const uint8_t* data, size_t size)
        : data_(data), size_(size), pos_(0), error_(false) {}

    bool ok()     const { return !error_; }

    // Read a single trivially-copyable value.
    template <typename T>
    bool read(T& val) {
        if (pos_ + sizeof(T) > size_) return fail();
        std::memcpy(&val, data_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return true;
    }

    // Read raw bytes.
    bool read_bytes(void* dest, size_t n) {
        if (pos_ + n > size_) return fail();
        std::memcpy(dest, data_ + pos_, n);
        pos_ += n;
        return true;
    }

    // Read a string previously written with write_string.
    bool read_string(std::string& s) {
        uint32_t len;
        if (!read(len)) return false;
        if (pos_ + len > size_) return fail();
        s.assign(reinterpret_cast<const char*>(data_ + pos_), len);
        pos_ += len;
        return true;
    }

    // Read a vector previously written with write_trivial_vector.
    template <typename T>
    bool read_trivial_vector(std::vector<T>& v) {
        uint32_t n;
        if (!read(n)) return false;
        v.resize(n);
        if (n > 0 && !read_bytes(v.data(), static_cast<size_t>(n) * sizeof(T))) return false;
        return true;
    }

private:
    bool fail() { error_ = true; return false; }
    const uint8_t* data_;
    size_t size_;
    size_t pos_;
    bool error_;
};

// Map a mangled type name back to the matching std::type_index for the types
// used by cosmosis log entries. Unknown names map to typeid(void).
const std::type_index& type_index_for_name(const std::string& mangled)
{
    static const std::map<std::string, std::type_index> map = {
        { typeid(int).name(),                    std::type_index(typeid(int))                    },
        { typeid(bool).name(),                   std::type_index(typeid(bool))                   },
        { typeid(double).name(),                 std::type_index(typeid(double))                 },
        { typeid(std::string).name(),            std::type_index(typeid(std::string))            },
        { typeid(cosmosis::complex_t).name(),    std::type_index(typeid(cosmosis::complex_t))    },
        { typeid(cosmosis::vint_t).name(),       std::type_index(typeid(cosmosis::vint_t))       },
        { typeid(cosmosis::vdouble_t).name(),    std::type_index(typeid(cosmosis::vdouble_t))    },
        { typeid(cosmosis::vstring_t).name(),    std::type_index(typeid(cosmosis::vstring_t))    },
        { typeid(cosmosis::vcomplex_t).name(),   std::type_index(typeid(cosmosis::vcomplex_t))   },
        { typeid(cosmosis::nd_int_t).name(),     std::type_index(typeid(cosmosis::nd_int_t))     },
        { typeid(cosmosis::nd_double_t).name(),  std::type_index(typeid(cosmosis::nd_double_t))  },
        { typeid(cosmosis::nd_complex_t).name(), std::type_index(typeid(cosmosis::nd_complex_t)) },
        { typeid(void*).name(),                  std::type_index(typeid(void*))                  },
    };
    static const std::type_index fallback(typeid(void));
    std::map<std::string, std::type_index>::const_iterator it = map.find(mangled);
    return it != map.end() ? it->second : fallback;
}

// Write one entry's payload (after the name and type tag have been written).
void write_entry(BinaryWriter& w, const cosmosis::Section& sec,
                 const std::string& name, datablock_type_t type)
{
    switch (type) {
    case DBT_INT: {
        int v = 0; sec.get_val(name, v);
        w.write(v);
        break;
    }
    case DBT_BOOL: {
        bool v = false; sec.get_val(name, v);
        w.write(static_cast<uint8_t>(v ? 1 : 0));
        break;
    }
    case DBT_DOUBLE: {
        double v = 0.0; sec.get_val(name, v);
        w.write(v);
        break;
    }
    case DBT_COMPLEX: {
        cosmosis::complex_t v; sec.get_val(name, v);
        double re = v.real(), im = v.imag();
        w.write(re); w.write(im);
        break;
    }
    case DBT_STRING: {
        std::string v; sec.get_val(name, v);
        w.write_string(v);
        break;
    }
    case DBT_INT1D: {
        cosmosis::vint_t v; sec.get_val(name, v);
        w.write_trivial_vector(v);
        break;
    }
    case DBT_DOUBLE1D: {
        cosmosis::vdouble_t v; sec.get_val(name, v);
        w.write_trivial_vector(v);
        break;
    }
    case DBT_COMPLEX1D: {
        // std::complex<double> is guaranteed layout-compatible with double[2].
        const cosmosis::vcomplex_t& v = sec.view<cosmosis::vcomplex_t>(name);
        w.write_trivial_vector(v);
        break;
    }
    case DBT_STRING1D: {
        const cosmosis::vstring_t& v = sec.view<cosmosis::vstring_t>(name);
        w.write(static_cast<uint32_t>(v.size()));
        for (size_t k = 0; k < v.size(); ++k) w.write_string(v[k]);
        break;
    }
    case DBT_INTND: {
        const cosmosis::nd_int_t& v = sec.view<cosmosis::nd_int_t>(name);
        w.write(static_cast<uint32_t>(v.ndims()));
        for (size_t k = 0; k < v.ndims(); ++k)
            w.write(static_cast<uint64_t>(v.extents()[k]));
        w.write(static_cast<uint32_t>(v.size()));
        if (v.size() > 0)
            w.write_bytes(&*v.cbegin(), v.size() * sizeof(int));
        break;
    }
    case DBT_DOUBLEND: {
        const cosmosis::nd_double_t& v = sec.view<cosmosis::nd_double_t>(name);
        w.write(static_cast<uint32_t>(v.ndims()));
        for (size_t k = 0; k < v.ndims(); ++k)
            w.write(static_cast<uint64_t>(v.extents()[k]));
        w.write(static_cast<uint32_t>(v.size()));
        if (v.size() > 0)
            w.write_bytes(&*v.cbegin(), v.size() * sizeof(double));
        break;
    }
    case DBT_COMPLEXND: {
        const cosmosis::nd_complex_t& v = sec.view<cosmosis::nd_complex_t>(name);
        w.write(static_cast<uint32_t>(v.ndims()));
        for (size_t k = 0; k < v.ndims(); ++k)
            w.write(static_cast<uint64_t>(v.extents()[k]));
        w.write(static_cast<uint32_t>(v.size()));
        if (v.size() > 0)
            w.write_bytes(&*v.cbegin(), v.size() * sizeof(cosmosis::complex_t));
        break;
    }
    default:
        break;
    }
}

// Read one entry's payload and insert it into the section.
bool read_entry(BinaryReader& r, cosmosis::Section& sec,
                const std::string& name, datablock_type_t type)
{
    switch (type) {
    case DBT_INT: {
        int v; if (!r.read(v)) return false;
        return sec.put_val(name, v) == DBS_SUCCESS;
    }
    case DBT_BOOL: {
        uint8_t b; if (!r.read(b)) return false;
        return sec.put_val(name, bool(b != 0)) == DBS_SUCCESS;
    }
    case DBT_DOUBLE: {
        double v; if (!r.read(v)) return false;
        return sec.put_val(name, v) == DBS_SUCCESS;
    }
    case DBT_COMPLEX: {
        double re, im;
        if (!r.read(re) || !r.read(im)) return false;
        return sec.put_val(name, cosmosis::complex_t(re, im)) == DBS_SUCCESS;
    }
    case DBT_STRING: {
        std::string v; if (!r.read_string(v)) return false;
        return sec.put_val(name, v) == DBS_SUCCESS;
    }
    case DBT_INT1D: {
        cosmosis::vint_t v; if (!r.read_trivial_vector(v)) return false;
        return sec.put_val(name, v) == DBS_SUCCESS;
    }
    case DBT_DOUBLE1D: {
        cosmosis::vdouble_t v; if (!r.read_trivial_vector(v)) return false;
        return sec.put_val(name, v) == DBS_SUCCESS;
    }
    case DBT_COMPLEX1D: {
        cosmosis::vcomplex_t v; if (!r.read_trivial_vector(v)) return false;
        return sec.put_val(name, v) == DBS_SUCCESS;
    }
    case DBT_STRING1D: {
        uint32_t n; if (!r.read(n)) return false;
        cosmosis::vstring_t v(n);
        for (uint32_t k = 0; k < n; ++k) if (!r.read_string(v[k])) return false;
        return sec.put_val(name, v) == DBS_SUCCESS;
    }
    case DBT_INTND: {
        uint32_t ndims; if (!r.read(ndims)) return false;
        std::vector<std::size_t> extents(ndims);
        for (uint32_t k = 0; k < ndims; ++k) {
            uint64_t ext; if (!r.read(ext)) return false;
            extents[k] = static_cast<std::size_t>(ext);
        }
        uint32_t n; if (!r.read(n)) return false;
        std::vector<int> data(n);
        if (n > 0 && !r.read_bytes(data.data(), static_cast<size_t>(n) * sizeof(int))) return false;
        return sec.put_val(name, cosmosis::nd_int_t(data, extents)) == DBS_SUCCESS;
    }
    case DBT_DOUBLEND: {
        uint32_t ndims; if (!r.read(ndims)) return false;
        std::vector<std::size_t> extents(ndims);
        for (uint32_t k = 0; k < ndims; ++k) {
            uint64_t ext; if (!r.read(ext)) return false;
            extents[k] = static_cast<std::size_t>(ext);
        }
        uint32_t n; if (!r.read(n)) return false;
        std::vector<double> data(n);
        if (n > 0 && !r.read_bytes(data.data(), static_cast<size_t>(n) * sizeof(double))) return false;
        return sec.put_val(name, cosmosis::nd_double_t(data, extents)) == DBS_SUCCESS;
    }
    case DBT_COMPLEXND: {
        uint32_t ndims; if (!r.read(ndims)) return false;
        std::vector<std::size_t> extents(ndims);
        for (uint32_t k = 0; k < ndims; ++k) {
            uint64_t ext; if (!r.read(ext)) return false;
            extents[k] = static_cast<std::size_t>(ext);
        }
        uint32_t n; if (!r.read(n)) return false;
        std::vector<cosmosis::complex_t> data(n);
        if (n > 0 && !r.read_bytes(data.data(), static_cast<size_t>(n) * sizeof(cosmosis::complex_t))) return false;
        return sec.put_val(name, cosmosis::nd_complex_t(data, extents)) == DBS_SUCCESS;
    }
    default:
        return false;
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// DataBlock::serialize / deserialize
// ---------------------------------------------------------------------------

std::vector<uint8_t> cosmosis::DataBlock::serialize() const
{
    std::vector<uint8_t> buf;
    BinaryWriter w(buf);

    w.write(DATABLOCK_SERIAL_MAGIC);
    w.write(DATABLOCK_SERIAL_VERSION);

    // --- sections ---
    w.write(static_cast<uint32_t>(sections_.size()));
    for (std::map<std::string,Section>::const_iterator si = sections_.begin();
         si != sections_.end(); ++si)
    {
        const std::string& sec_name = si->first;
        const Section&     sec      = si->second;
        w.write_string(sec_name);

        size_t nvals = sec.number_values();
        w.write(static_cast<uint32_t>(nvals));
        for (size_t j = 0; j < nvals; ++j) {
            const std::string& name = sec.value_name(j);
            w.write_string(name);
            datablock_type_t type = DBT_UNKNOWN;
            sec.get_type(name, type);
            w.write(static_cast<uint32_t>(type));
            write_entry(w, sec, name, type);
        }
    }

    // --- access log ---
    w.write(static_cast<uint32_t>(access_log_.size()));
    for (size_t i = 0; i < access_log_.size(); ++i) {
        const log_entry& e = access_log_[i];
        w.write_string(std::get<0>(e));           // log_type
        w.write_string(std::get<1>(e));           // section
        w.write_string(std::get<2>(e));           // name
        w.write_string(std::get<3>(e).name());    // mangled C++ type name
    }

    return buf;
}

bool cosmosis::DataBlock::deserialize(const std::vector<uint8_t>& blob)
{
    BinaryReader r(blob.data(), blob.size());

    uint64_t magic;
    if (!r.read(magic) || magic != DATABLOCK_SERIAL_MAGIC) return false;

    uint32_t version;
    if (!r.read(version) || version != DATABLOCK_SERIAL_VERSION) return false;

    // Parse into temporary containers so that on failure the original
    // contents of sections_ and access_log_ are left intact.
    std::map<std::string, Section> tmp_sections;
    std::vector<log_entry> tmp_log;

    // --- sections ---
    uint32_t nsections;
    if (!r.read(nsections)) return false;

    for (uint32_t i = 0; i < nsections; ++i) {
        std::string sec_name;
        if (!r.read_string(sec_name)) return false;
        Section& sec = tmp_sections[sec_name];

        uint32_t nvals;
        if (!r.read(nvals)) return false;
        for (uint32_t j = 0; j < nvals; ++j) {
            std::string name;
            if (!r.read_string(name)) return false;
            uint32_t type_raw;
            if (!r.read(type_raw)) return false;
            datablock_type_t type = static_cast<datablock_type_t>(type_raw);
            if (!read_entry(r, sec, name, type)) return false;
        }
    }

    // --- access log ---
    uint32_t nlog;
    if (!r.read(nlog)) return false;
    tmp_log.reserve(nlog);
    for (uint32_t i = 0; i < nlog; ++i) {
        std::string log_type, section, name, type_name;
        if (!r.read_string(log_type)) return false;
        if (!r.read_string(section))  return false;
        if (!r.read_string(name))     return false;
        if (!r.read_string(type_name)) return false;
        tmp_log.push_back(log_entry(log_type, section, name,
                                    type_index_for_name(type_name)));
    }

    if (!r.ok()) return false;

    sections_ = std::move(tmp_sections);
    access_log_ = std::move(tmp_log);
    return true;
}

bool cosmosis::DataBlock::serialize(const std::string& path) const
{
    std::vector<uint8_t> blob = serialize();
    if (blob.empty()) return false;
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(blob.data()),
            static_cast<std::streamsize>(blob.size()));
    return f.good();
}

bool cosmosis::DataBlock::deserialize(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    std::streamsize size = f.tellg();
    if (size < 0) return false;
    f.seekg(0);
    std::vector<uint8_t> blob(static_cast<std::size_t>(size));
    f.read(reinterpret_cast<char*>(blob.data()), size);
    if (!f) return false;
    return deserialize(blob);
}
