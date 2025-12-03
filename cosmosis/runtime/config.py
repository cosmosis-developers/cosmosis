#coding: utf-8


u"""Definition of the :class:`Inifile` class."""


import os
import sys
import collections
import configparser
import io

class CosmosisConfigurationError(configparser.Error):
    u"""Something to throw when there is an error in a .ini file particular to Cosmosis.

    The (underlying) object simply carries a string providing information
    to the user.

    """
    pass

class IncludingConfigParser(configparser.ConfigParser):
    u"""Extension of built-in python :class:`ConfigParser` to %include other files.

    Use the line: %include filename.ini This is assumed to end a section,
    and the last section in the included file is assumed to end as well

    Note that the caller of the `read()` method may set a
    `no_expand_includes` attribute on this object, to cause any %include
    lines to *not* actually be actioned (they will be regarded as
    comments, but still delineate sections).
    """

    def __init__(self, defaults=None, print_include_messages=True, no_expand_vars=False):
        self.no_expand_vars = no_expand_vars
        configparser.ConfigParser.__init__(self,
                                   defaults=defaults,
                                   dict_type=collections.OrderedDict,
                                   strict=False,
                                   inline_comment_prefixes=(';', '#'),
                                   )
        self.print_include_messages = print_include_messages

    def _read(self, fp, fpname):
        """
        This overrides the parent method to allow %include directives
        to import additional files.

        To do so we first read the file into a StringIO object, dealing
        with the %include directives as we go, then pass that to the
        parent method.
        """
        s = io.StringIO()
        for line in fp:
            # check for include directives
            if not self.no_expand_vars:
                line = os.path.expandvars(line)

            if line.lower().startswith('%include'):
                _, filename = line.split()
                filename = filename.strip('"').strip("'")

                if self.print_include_messages:
                    print(f"Reading included ini file: {filename}")
                if not os.path.exists(filename):
                    raise ValueError(f"Tried to include non-existent file {filename}")

                # read the contents of the ini file into a new instance
                # of this class, then we will write it out
                sub_ini = self.__class__(filename)

                # write the whole other file content to our StringIO
                sub_ini.write(s)
            else:
                s.write(line)

        # rewind the stuff we have read
        s.seek(0)
        return super()._read(s, fpname)



class Inifile(IncludingConfigParser):

    u"""A dictionary of `(section, name) -> value` pairs corresponding to entries in a .ini file.

    The class is designed to hide the details of parsing .ini files, and
    then for creating :class:`DataBlock` objects (which wrap C objects
    and can be passed down a processing pipeline which may include modules
    written in C) via the :class:`Pipeline` and then :class:`Module`
    constructors (see `Pipeline.__init__()`).

    The values are all stored as strings, with methods provided to locate
    and interpret them as integers, floating-point numbers, booleans, or
    arrays of integers or floating-point numbers.

    """

    def __init__(self, filename, defaults=None, override=None, print_include_messages=True, no_expand_vars=False):
        u"""Read in a configuration from `filename`.

        The `defaults` will be applied if a parameter is not specified in
        the file (or %included descendants), and the `override`s will be
        imposed on the regardless of whether those parameters have
        assigned values or not.

        Where supplied, `defaults` and `override` should be dictionary
        mappings of `(section, name) -> value`.

        """

        IncludingConfigParser.__init__(self,
                                       defaults=defaults,
                                       print_include_messages=print_include_messages,
                                       no_expand_vars=no_expand_vars)

        # if we are pased a dict, convert it to an inifile
        if isinstance(filename, dict):
            for section, values in filename.items():
                self.add_section(section)
                for key, value in values.items():
                    self.set(section, key, str(value))
        elif isinstance(filename, Inifile):
            # This seems to be the only way to preserve the
            # defaults.
            # https://stackoverflow.com/questions/23416370/manually-building-a-deep-copy-of-a-configparser-in-python-2-7
            s = io.StringIO()
            filename.write(s)
            s.seek(0)
            self.read_file(s)
        elif hasattr(filename, "read"):
            self.read_file(filename)
        # default read behaviour is to ignore unreadable files which
        # is probably not what we want here
        elif filename is not None:
            if isinstance(filename,str) and not os.path.exists(filename):
                raise IOError("Unable to open configuration file `" + filename + "'")
            self.read(filename)

        # override parameters
        if override:
            for section, name in override:
                if section=="DEFAULT":
                    self._defaults[name] = override[(section,name)]
                else:
                    if not self.has_section(section):
                        self.add_section(section)
                    self.set(section, name, override[(section, name)])

    @classmethod
    def from_lines(cls, lines, *args, **kwargs):
        u"""Create an Inifile from a list of lines."""
        s = io.StringIO("\n".join(lines))
        return cls(s, *args, **kwargs)

    def __iter__(self):
        u"""Iterate over all the parameters.

        The value of the iterator is `((section, name), value)`.

        """
        return (((section, name), value) for section in self.sections()
                for name, value in self.items(section))



    def items(self, section, raw=False, vars=None, defaults=True):
        u"""Return a list of pairs (key, value) from all the options in a given `section`.

        If raw is set, do not replace values which are set using the ini file 
        interpolation syntax %(name)s.

        If vars is set to a dictionary, use it as an additional source of options.

        If defaults is True (the default), parameters in the [DEFAULT] section are included
        in all other sections.


        """
        if defaults:
            return IncludingConfigParser.items(self, section, raw=raw, vars=vars)
        else:
            d = collections.OrderedDict()
            try:
                d.update(self._sections[section])
            except KeyError:
                if section != configparser.DEFAULTSECT:
                    raise configparser.NoSectionError(section)
            # Update with the entry specific variables
            if vars:
                for key, value in list(vars.items()):
                    d[self.optionxform(key)] = value
            options = list(d.keys())
            if raw:
                return [(option, d[option])
                        for option in options]
            else:
                return [(option, self._interpolate(section, option, d[option], d))
                        for option in options]


    def get(self, section, option, raw=False, vars=None, fallback=configparser._UNSET):

        u"""Get a value as a string, or `default` if the value is not in the dictionary.

        If the `default` is not set and is needed, an error with a message
        to the user will be raised. (`None` is *not* acceptable as a
        default).

        """
        try:
            return IncludingConfigParser.get(self, section, option, raw=raw, vars=vars, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            if fallback is configparser._UNSET:
                raise CosmosisConfigurationError("CosmoSIS looked for an option called '%s' in the '[%s]' section, but it was not in the ini file"%(option,section))
            else:
                return fallback
            
    def __getitem__(self, key: tuple):
        section, option = key
        return self.get(section, option)

    def __setitem__(self, key: tuple, value: str):
        section, option = key
        self.set(section, option, str(value))

    # these functions override the default parsers to allow for extra formats
    def getint(self, section, option, raw=False, vars=None, fallback=configparser._UNSET):
        u"""Get a value as an integer, or return `default` if the value is not found.
        
        If the `default` is not set and is needed, an error with a message
        to the user will be raised. (`None` is *not* acceptable as a
        default).

        """
        try:
            return IncludingConfigParser.getint(self, section, option, raw=raw, vars=vars, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError, CosmosisConfigurationError) as e:
            if fallback is configparser._UNSET:
                raise CosmosisConfigurationError("CosmoSIS looked for an integer option called '%s' in the '[%s]' section, but it was not in the ini file"%(option,section))
            elif not isinstance(fallback, int):
                raise TypeError("Default not integer")
            else:
                return fallback

    def getfloat(self, section, option, raw=False, vars=None, fallback=configparser._UNSET):
        u"""Get a floating-point value from the dictionary, with `default`.

        If the value is not found in the dictionary and `default` is
        specified, then `default` will be returned.  Otherwise an error
        will be thrown with a useful message for the user.

        """
        try:
            return IncludingConfigParser.getfloat(self, section, option, raw=raw, vars=vars, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError, CosmosisConfigurationError) as e:
            if fallback is configparser._UNSET:
                raise CosmosisConfigurationError("CosmoSIS looked for a float option called '%s' in the '[%s]' section, but it was not in the ini file"%(option,section))
            elif not isinstance(fallback, float):
                raise TypeError("Default not float")
            else:
                return fallback

    def getboolean(self, section, option, raw=False, vars=None, fallback=configparser._UNSET):
        u"""Interpret a parameter as a boolean, including symbolic values.

        This essentially allows a configuration file to represent boolean
        values in the most convenient manner (‘true’, ‘n’, etc) as well as
        with zero/non-zero numerical values.

        If the parameter is not found in the dictionary, then `default`
        will be returned, which will itself default to `False` if not
        specified.

        """
        try:
            return IncludingConfigParser.getboolean(self, section, option, raw=raw, vars=vars, fallback=fallback)
        except ValueError:
            # additional options t/y/n/f
            value = self.get(section, option).lower()
            if value in ['y', 'yes', 't','true']:
                return True
            elif value in ['n', 'no', 'f', 'false']:
                return False
            else:
                raise ValueError("Unable to parse parameter "
                                 "%s--%s = %s into boolean form"
                                 % (section, option, value))
        except (configparser.NoSectionError, configparser.NoOptionError, CosmosisConfigurationError) as e:
            if fallback is configparser._UNSET:
                raise CosmosisConfigurationError("CosmoSIS looked for a boolean (T/F) option called '%s' in the '[%s]' section, but it was not in the ini file"%(option,section))
            elif not isinstance(fallback, bool):
                raise TypeError("Default not boolean")
            else:
                return fallback



    def gettyped(self, section, name):
        u"""Best-guess the type of a parameter and return it as that type.

        The method will try parsing the value as, in this order:
            an integer or list of integers,
            a float or list of floats,
            a complex number or list of complex numbers,
            a boolean,
            a string.

        The string value is the fallback if all else fails.
        """

        import re

        value = IncludingConfigParser.get(self, section, name)
        value = value.strip()
        # There isn't really a sensible thing to return for this,
        # so we just need to set it to None.
        if not value: return None
        # try quoted string
        m = re.match(r"^(['\"])(.*?)\1$", value)
        if m is not None:
            return m.group(2)

        value_list = value.split()

        # Try to match integer array.  This will fail whenever a decimal
        # point occurs anywhere in the list of values.
        try:
            parsed = [int(s) for s in value_list]
            if len(parsed) == 1:
                return parsed[0]
            else:
                return parsed
        except ValueError:
            pass

        # try to match float array
        try:
            parsed = [float(s) for s in value_list]
            if len(parsed) == 1:
                return parsed[0]
            else:
                return parsed
        except ValueError:
            pass

        # try to match complex array
        try:
            parsed = [complex(s) for s in value_list]
            if len(parsed) == 1:
                return parsed[0]
            else:
                return parsed
        except ValueError:
            pass

        # try to match boolean (no array support)

        try:
            return self.getboolean(section, name)
        except ValueError:
            pass

        # default to string
        return value
