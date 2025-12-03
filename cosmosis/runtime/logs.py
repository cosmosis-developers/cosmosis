import logging
import sys
import os

NOISY = 15


# Change the logger so it doesn't print out all the boilerplate, unless requested
logger = logging.getLogger("cosmosis")
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)

# The pytest tests about whether a particular log message is printed
# fail if the logger is not set to propagate, I think because it replaces
# the logger. So we only set it to not propagate if we are running
# the tests, not otherwise. I'm probably using this wrong and should just
# not use the python logging system, it's not really a good fit for this
# application
if ("PYTEST_VERSION" in os.environ) or ("COSMOSIS_TESTING" in os.environ):
    logger.propagate = True
else:
    logger.propagate = False


verbosity_levels = {
    "debug": 40,    #40 log level debug
    "noisy": 35,               #35 log level noisy
    "standard": 30,  #30 log level info
    "normal": 30,  #30 log level info
    "quiet": 20, #20 log level warning / overview
    "muted": 10,    #10 log level error / important
    "silent": -1,   #-1
}


def set_verbosity(verb):
    try:
        verb = int(verb)
    except ValueError:
        pass
    if not isinstance(verb, int):
        try:
            verb = verbosity_levels[verb]
        except KeyError:
            valid_levels = ', '.join(list(verbosity_levels.keys()))
            message = """Error specifiying verbosity.
                You put: '{0}'.
                We want either an integer 0 (silent) - 50 (everything) 
                or one of: {1}""".format(verb, valid_levels)
            raise ValueError(message)
    level = 50 - verb
    debug(f"CosmoSIS verbosity set to {verb}")
    set_level(level)

def set_level(level):
    logger.setLevel(level)
    handler.setLevel(level)

def get_level():
    return logger.getEffectiveLevel()

def enable_log_tracing():
    formatter = logging.Formatter('[%(pathname)s Line %(lineno)d]: %(message)s')
    handler.setFormatter(formatter)

# level 10
def debug(message):
    logger.log(logging.DEBUG, message)
# extreme level debugging information, printing basically everything

# level 15
def noisy(message):
    logger.log(NOISY, message)

# level 20
def info(message):
    logger.log(logging.INFO, message)

# level 30
def warning(message):
    logger.log(logging.WARNING, message)

def overview(message):
    logger.log(logging.WARNING, message)

# Level 40
def error(message):
    logger.log(logging.ERROR, message)

def important(message):
    logger.log(logging.ERROR, message)

# Level 50
def critical(message):
    logger.log(logging.CRITICAL, message)

def is_enabled_for(level):
    return logger.isEnabledFor(level)
