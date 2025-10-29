from .block import DataBlock, BlockError
from .import section_names as names
import os

try:
	from .lib import enable_cosmosis_segfault_handler
except:
	_allow_unbuilt_import = os.environ.get("COSMOSIS_ALLOW_UNBUILT_IMPORT", "0") == "1"
	if not _allow_unbuilt_import:
		raise
