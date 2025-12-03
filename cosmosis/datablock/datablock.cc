#include "datablock.hh"
#include "clamp.hh"
#include <iostream>
#include "cxxabi.h"
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
