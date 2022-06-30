MorpholoPy
==========

Rewrite of the morphology pipeline of https://github.com/correac/morpholopy.

Usage
=====

The pipeline consists of an executable script, `morphology-pipeline` that is very similar to the 
`swift-pipeline` script. There is currently no easy way to install this script, so you simply have to run it 
from within its own directory.

Usage:
```
./morphology-pipeline \
  -C <configuration directory> \
  -i <input folder that contains the snapshots and catalogue> \
  -s <input snapshot> \
  -c <input catalogue> \
  -n <name for the run> \
  -g <number of galaxies for which individual plots are made> \
  -o <directory where output images and web pages are stored> \
  -M <stellar mass (in Msun) above which halos are included> \
  -j <number of parallel processes to use>
```

The configuration directory should contain a `description.html` file that will be used as a header for the web 
page (as in the normal pipeline), and a `config.yml` file containing the plots that will be included. This 
uses (_hacks_ is a better description) the normal pipeline functionalities, so the syntax is not (yet?) ideal.
Example configuration file:
```
description_template: description.html

scripts:
  - section: HI size
    caption: HI size plot
    output_file: HI_size_mass.png
    title: HI size
```

The pipeline will output some plots and a web page, plus a `.yml` file
(`morphology_data_<snapshot number>.yml`) that contains a lot of data and that can be used for comparisons.
However, comparisons are currently not yet supported.
