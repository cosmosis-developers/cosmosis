#!/usr/bin/env python
import sys
import argparse
import os
from .postprocessing import run_cosmosis_postprocess

parser = argparse.ArgumentParser(description="Post-process cosmosis output")
parser.add_argument("inifile", nargs="+")
mcmc=parser.add_argument_group(title="MCMC", description="Options for MCMC-type samplers")
mcmc.add_argument("--burn", default=0.0, type=float, help="Fraction or number of samples to burn at the start")
mcmc.add_argument("--thin", default=1, type=int, help="Keep every n'th sampler in MCMC")
mcmc.add_argument("--weights", action='store_true', help="Look for a weight column in a generic MCMC file")
mcmc.add_argument("-gd","--getdist", action='store_true', default=False, help="Use getdist for the statistics of MCMC chain. Not implemented for polychord and grid sampler yet.")

general=parser.add_argument_group(title="General", description="General options for controlling postprocessing")
general.add_argument("-o","--outdir", default=".", help="Output directory for all generated files")
general.add_argument("-p","--prefix", default="", help="Prefix for all generated files")
general.add_argument("--more-latex", default="", help="Load an additional latex file to the default")
general.add_argument("--no-latex", action='store_true', help="Do not use latex-style labels, just use the text")
general.add_argument("--blind-add", action='store_true', help="Blind results by adding adding a secret value to each parameter")
general.add_argument("--blind-mul", action='store_true', help="Blind results by scaling by a secret value for each parameter")
general.add_argument("--pdb", action='store_true', help="Run the debugger if any of the postprocessing stages fail")
general.add_argument("--fatal-errors", action='store_true', help="Errors are fatal; useful for testing only")

inputs=parser.add_argument_group(title="Inputs", description="Options controlling the inputs to this script")
inputs.add_argument("--text", action='store_true', help="Tell postprocess that its argument is a text file, regardless of its suffix")
inputs.add_argument("--derive", default="", help="Read a python script with functions in that derive new columns from existing ones")

plots=parser.add_argument_group(title="Plotting", description="Plotting options")
plots.add_argument("--legend", help="Add a legend to the plot with the specified titles, separated by | (the pipe symbol)")
plots.add_argument("--legend-loc", default='best', help="The location of the legend: best, UR, UL, LL, LR, R, CL, CR, LC, UC, C (use quotes for the ones with two words.)")
plots.add_argument("--swap", action='store_true', help="Swap the ordering of the parameters in (x,y)")
plots.add_argument("--only", type=str, dest='prefix_only', help="Only make 2D plots where both parameter names start with this")
plots.add_argument("--either", type=str, dest='prefix_either', help="Only make 2D plots where one of the parameter names starts with this.")
parser.add_argument("--exclude", nargs="+", dest='prefix_exclude', help='Specify one or more prefixes to exclude matching parameters from 2D plots')

plots.add_argument("--no-plots", action='store_true', help="Do not make any default plots")
plots.add_argument("--no-2d", action='store_true', help="Do not make any 2D plots")
plots.add_argument("--no-alpha", dest='alpha', action='store_false', help="No alpha effect - shaded contours will not be visible through other ones")
plots.add_argument("-f", "--file-type", default="png", help="Filename suffix for plots")
plots.add_argument("--no-smooth", dest='smooth', default=True, action='store_false', help="Do not smooth grid plot joint constraints")
plots.add_argument("--fix-edges", dest='fix_edges', default=True, action='store_true', help="Use an alternative KDE to fix 1D plot boundaries")
plots.add_argument("--no-fix-edges", dest='fix_edges', default=False, action='store_false', help="Switch off the edge fixing")
plots.add_argument("--n-kde", default=100, type=int, help="Number of KDE smoothing points per dimension to use for MCMC 2D curves. Reduce to speed up, but can make plots look worse.")
plots.add_argument("--factor-kde", default=2.0, type=float, help="Smoothing factor for MCMC plots.  More makes plots look better but can smooth out too much.")
plots.add_argument("--no-fill", dest='fill', default=True, action='store_false', help="Do not fill in 2D constraint plots with color")
plots.add_argument("--extra", dest='extra', default="", help="Load extra post-processing steps from this file.")
plots.add_argument("--tweaks", dest='tweaks', default="", help="Load plot tweaks from this file.")
plots.add_argument("--no-image", dest='image', default=True, action='store_false', help="Do not plot the image in  2D grids; just show the contours")
plots.add_argument("--run-max-post", default="", help="Run the test sampler on maximum-posterior sample and save to the named directory.")
plots.add_argument("--truth", default="", help="An ini file containing truth values to mark on plots")


def main(args):
	args = parser.parse_args(args)

	# Run the main postprocessing code
	processor = run_cosmosis_postprocess(args.inifile, **vars(args))

	# The postprocessor doesn't finalize the plots until it is saved
	# so that we can use it interactively too. So save here.
	if processor is not None:
		#Save all the image files and close the text files
		processor.save()

if __name__=="__main__":
	main(sys.argv[1:])
