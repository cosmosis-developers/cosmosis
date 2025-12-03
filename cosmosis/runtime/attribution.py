import yaml
import collections
import os

#  Tasks:
# when we run a pipeline, collect all the attribution
# information for it.  Collate into one object.
# Provide a series of comment  lines

class ModuleAttribution(object):

	def __init__(self, data):
		self.data = data
		self.name = data.get('name', 'Unknown')
		self.version = data.get('version', '')
		self.attribution = self.to_list(
			data.get("attribution", "Unknown"))
		self.cite = self.to_list(
			data.get("cite", []))
	@staticmethod
	def to_list(obj):
		if isinstance(obj, list): return obj
		return [obj]

	@classmethod
	def from_yaml(cls,filename):
		if os.path.exists(filename):
			data = yaml.safe_load(open(filename))
		else:
			data = {}
		return cls(data)


class PipelineAttribution(object):
	def __init__(self, modules):
		#for each module, find the associated YAML file,
		#if there is one
		self.attributions = collections.OrderedDict()

		for module in modules:
			# we really don't ever want to cause a crash due to
			# an atrribution failure, so wrap all of this in try/except
			try:
				directory, _ = os.path.split(module.filename)
				filename = os.path.join(directory, "module.yaml")
				self.attributions[module.name] = ModuleAttribution.from_yaml(filename)
			except:
				continue

	def write_output(self, output):
		comments = []
		for i,(name, info) in enumerate(self.attributions.items()):
			comment = ""
			out_name = info.name
			if info.name=='Unknown':
				out_name = name
				comment = ("name from ini file")
			if info.version:
				out_name += ' ' + str(info.version)
			output.metadata("module_%d"%i, out_name, comment)
			for c in info.cite:
				output.comment("CITE "+c)


