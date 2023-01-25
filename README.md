MorpholoPy
==========

Rewrite of the morphology pipeline of https://github.com/correac/morpholopy.

Requirements
============

All required packages are listed in `requirements.txt`. The easiest way to get ready to use the 
pipeline is by setting up a virtual environment, like this:

```
> python3 -m venv build_env
> ./build_env/bin/pip install --upgrade pip
> ./build_env/bin/pip install -r requirements.txt
```

You can then run the morphology pipeline using

```
./build_env/bin/python3 morphology-pipeline
```

We will simply refer to this command as `./morphology-pipeline` below.

Usage
=====

The pipeline consists of an executable script, `morphology-pipeline` that is very similar to 
the `swift-pipeline` script. There is currently no easy way to install this script, so you 
simply have to run it from within its own directory.

Usage:
```
./morphology-pipeline \
  -C/--config      <configuration directory> \
  -i/--input       <input folder that contains the snapshots and catalogues> \
  -s/--snapshots   <input snapshot> \
  -c/--catalogues  <input catalogue> \
  -o/--output      <directory where output images and web pages are stored> \
 [-n/--run-names   <name for the run>] \
 [-j/--num-of-cpus <number of parallel processes to use>] \
 [-m/--metadata    <prefix for the meta-data output file] \
 [-d/--debug] \
 [-l/--lazy]
```
(the last four arguments are optional). `--debug` outputs additional debugging information, while
`--lazy` simply reads an existing meta-data file and creates the plots and web pages, without
recomputing any quantities. This is the same behaviour as the comparison mode, but for a single run.

To run in comparison mode, simply provide multiple arguments to `--input`, `--snapshots`, 
`--catalogues` and `--run-names`.

The configuration directory should contain a `description.html` file that will be used as a 
header for the web page (as in the normal pipeline), and a `config.yml` file containing 
additional configuration options. This uses (_hacks_ is a better description) the normal 
pipeline functionalities,. Example configuration file (containing default values for all 
variables):

``` 
# directory where observational data are stored,
# relative to the confiuration directory
observational_data_directory: ../velociraptor-comparison-data

# template HTML file for the simulation meta-data header on the web page,
# relative to the configuration directory
description_template: description.html

# matplotlib stylesheet to use,
# relative to the configuration directory
matplotlib_stylesheet: mnras.mplstyle


# parameters for selection of galaxies:
# only galaxies satisfying a stellar and gas mass selection are analysed

# minimum stellar mass for galaxies
# properties are only computed for galaxies above this mass
mass_limit_stars_in_Msun: 1.e6
mass_variable_stars: apertures.mass_star_30_kpc

# minimum gas mass for galaxies
# properties are only computed for galaxies above this mass
mass_limit_gas_in_Msun: 0.
mass_variable_gas: apertures.mass_gas_30_kpc

# parameters for the selection of galaxies that are plotted
# on an individual basis
# only galaxies within a mass range are eligible for plotting
# the number of galaxies that gets plotted is limited
# if the number of eligible galaxies is larger, we select a
# random subset
plotting_lower_mass_limit_in_Msun: 1.e10
plotting_upper_mass_limit_in_Msun: 1.e12
plotting_number_of_galaxies: 10
plotting_random_seed: 42

# the mass distribution perpendicular to the disk
# is binned for the exponential fit to calculate the
# scaleheight; the region used for the fit has a size
# of 4 stellar half mass radii (from -2 Rhalf to +2 Rhalf);
# the number of bins for each galaxy varies
# so that for each galaxy the bin size is smaller or 
# equal to scaleheight_binsize_kpc
scaleheight_binsize_kpc: 0.02

# minimum mass to attempt fitting the scaleheight
scaleheight_lower_mass_limit_in_Msun: 1.e7

# method used to determine the axis for face-on and edge-on projections:
# string consisting of:
#  <component type>_<inner mask radius>_<outer mask radius>_<sigma clipping>
# where:
#  - component type can be stars, gas, ISM, HI, baryons
#  - inner mask radius can be 0xR0.5, 0.5R0.5
#  - outer mask radius can be R0.5, 2xR0.5, 4xR0.5, 50kpc, R200crit, 0.1Rvir
#  - sigma clipping can be 0sigma (no clipping), 1sigma, 2sigma
# (R0.5 is the stellar half mass radius within 50kpc; sigma clipping is performed
# on the values of the angular momentum of the selected component before computing
# the total angular momentum vector
orientation_method: stars_0xR0.5_R0.5_0sigma
```

The pipeline will output some plots and a web page, plus a `.yml` file 
(`morphology_data_<snapshot number>.yml`) that contains a lot of data and that can be used for 
comparisons.

Adding new features
===================

In order to add new features to the pipeline, it is important to know how the pipeline works.
This is a six step process:
 1. Read the galaxy catalogue and select galaxies that need to be analysed.
 2. Loop over the galaxies (in parallel) and perform the analysis per galaxy. Output galaxy plots
    if this galaxy was selected for plotting. Add integrated galaxy properties to the global list
    of properties. Add resolved galaxy properties (pixel values, angular bin values) to global
    bins.
 3. Convert the global bins for resolved properties into median lines.
 4. Create a meta-data file containing all the global galaxy data and the median lines.
 5. Read the meta-data file (for all simulations if run in comparison mode).
 6. Create global plots of galaxy quantities and combined resolved quantities. Create the web 
    page.

When run in comparison mode, only steps 5 and 6 are performed.

For step 2 to work efficiently, it is important that we know from the start how much data will 
be generated. For integrated quantities, this is easy: every galaxy will add exactly one value 
for each quantity of interest. For resolved quantities, things are less clear. Spatially 
resolved surface density maps for example use a fixed physical pixel size, which means that the 
number of pixels can vary from galaxy to galaxy and is impossible to predict ahead of time. 
Even if predicting the number of pixels was possible, creating a buffer to store all the pixel 
values to then compute the medians in step 3 would require a lot of memory, which would 
futhermore scale with the number of galaxies. This would lead to problems with large 
simulations. A possible solution (used by the old morphology pipeline) is to output the 
resolved quantities to files on disk, and then read these files when creating the plots. While 
this works, it does produce a lot of output and will lead to a large number of disk operations, 
which are not very efficient.

To overcome these issues, we use a different approach to combine resolved galaxy data into 
median lines: we simply create a 2D histogram of the data values for each plot of interest. 
This is motivated by the fact that we already know the range we want to plot, the number of 
bins we want to use for the median line, and whether or not we want to use logarithmic scaling 
for the x and y axis. We can then easily set up a grid in x and y that covers this range.

For each cell in the grid, we can then count the number of resolved galaxy points that fall 
into that cell. At the end of the analysis, each column corresponding to a fixed x will contain 
a histogram of the values in y that fall within that x bin. We can easily find the median (and 
any other percentile of interest) from this histogram, with an accuracy of approximately the y 
bin width.

The number of bins in x is usually rather small (we use 20 for all plots right now). The number 
of bins in y will determine the accuracy with which we can compute the median: for 100 bins, 
this accuracy will be approximately 1% of the full y range of the plot. For visualisation, this 
is more than sufficient. We can then store all the information required to compute the median 
for any number of galaxies with any number of pixels or angular bins using a buffer with only 
2000 elements. This is very small in memory, and - more importantly - does no longer scale with 
the number of galaxies. To account for the contribution from a single galaxy, we simply add the 
2000 element array for that galaxy to the global 2000 element array, which is a very efficient 
operation.

With the above method to compute medians, we can predict the size of the global galaxy property 
buffer after step 1. We then create this buffer as a `numpy.array` with a structured data type. 
This type of array is in some ways similar to a dictionary, but has all the efficiency 
advantages of an actual numpy array. It allows you to have elements of different data types in 
a single array, and allows you to index elements or columns using a key, just like in a 
dictionary. Since an example might be more clear:

```
>>> import numpy as np
>>> arr = np.zeros(3, dtype=[("a", np.float32), ("b", np.int32), ("c", "U10")])
>>> arr["a"] = np.logspace(-2., 5., 3)
>>> arr["b"] = np.arange(3)
>>> arr["c"] = "value"
>>> print(arr)
[(9.9999998e-03, 0, 'value') (3.1622776e+01, 1, 'value')
 (1.0000000e+05, 2, 'value')]
>>> for row in arr:
...   print(row)
... 
(0.01, 0, 'value')
(31.622776, 1, 'value')
(100000., 2, 'value')
>>> for row in arr:
...   print(row["b"])
... 
0
1
2
>>> print(arr["a"])
[9.9999998e-03 3.1622776e+01 1.0000000e+05]
```

What is even more useful, is the fact that structured array columns can be arrays themselves:

```
>>> import numpy as np
>>> arr = np.zeros(3, dtype=[("a", np.int32), ("b", (np.float32, 3))])
>>> print(arr)
[(0, [0., 0., 0.]) (0, [0., 0., 0.]) (0, [0., 0., 0.])]
>>> print(arr["b"])
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]
>>> print(arr["b"].shape)
(3, 3)
```

In `morpholopy/galaxy_data.py`, we maintain a list of all the quantities we want to compute for 
every galaxy, and its data type. This data type can be a basic data type, but also an array of 
such types (useful for storing positions and vectors). This is essentially a large data type 
list that can be passed on to the `dtype` argument of a `numpy.array` or `numpy.zeros` 
function. After we computed the number of galaxies that will be analysed, we initialise the 
global array with this number and data type.

When processing a single galaxy, we also construct such an array, but then with one element. We 
perform the various analyses and set the corresponding elements of the galaxy data array to 
their appropriate values. Finally, we return the galaxy array and use it to update the relevant 
row in the global array. This is all handled by `numpy`, and is therefore as efficient as it 
can get.

This also has important implications for the parallelisation. For this, we use the 
`multiprocessing` module, using the `forkserver` initialisation method for subprocesses. In 
this mode, parallel subprocesses spawned by the main pipeline process do not inherit any 
resources that are not required, i.e. they do not automatically get access to the variables 
(and memory contents) of the main process. This is different from the default way 
`multiprocessing` creates subprocesses, where each subprocess gets a full copy of all the 
variables of the main process at the time the process is created. In other words, the 
subprocesses that are responsible for the analysis of individual galaxies do not have access to 
the global galaxy property array, but also do not need to store a copy of it in memory, which 
significantly reduces their memory footprint. They instead receive a minimal amount of 
information from the main process at the start of each analysis (i.e. the catalogue properties 
of an individual galaxy, the galaxy index and some control variables) and communicate back the 
galaxy index and the galaxy data array at the end. The main process then efficiently sets the 
corresponding row of the global array with minimal inter-process communication.

To add a new variable to the pipeline, you will need to
 1. Create an appropriate entry at the top of `galaxy_data.py`. For integrated quantities,
    just add an extra element to the `data_fields` list, for resolved quantities, add a
    new element to the `medians` dictionary, using the appropriate information about the
    median plotting range, number of bins, units and plotting labels.
 2. Implement the calculation in a new file or one of the existing files. Calculations are
    roughly ordered in logical categories, i.e. `morphology.py` contains the calculation of
    axis lengths and angular momenta, `KS.py` contains Kennicutt-Schmidt like surface density
    calculations, and `HI_size.py` contains HI size calculations.
 3. Call the new calculation function from the `process_galaxy` function in `galaxy_data.py`.
    Make sure to use the already computed `face_on_rmatrix` and `edge_on_rmatrix` that use
    the appropriate (and consistent) galaxy orientation if you need any projections.
    Note that all gas and star particles have already been recentred onto the galaxy centre of
    potential and centre of velocity. The `data` object is a `swiftsimio.SWIFTDataset` with all
    of its functionality. You are strongly encouraged to use existing `swiftsimio` 
    functionality where appropriate.
 4. Implement a plotting routine for the new quantity. Plotting routines are put in the same 
    file where the corresponding calculations are implemented for clarity. Plotting routines 
    should always assume a list of global galaxy property arrays as input, i.e. they are used 
    for both the single galaxy mode and the comparison mode. Plotting routines should try to 
    make use of the already provided plotting functions in `plot.py`, or the median specific
    plotting routines in `medians.py`. If no appropriate plotting routine exists, you might 
    want to consider adding one in one of these files. Plotting routines should more or less
    stick to the following API:
    ```
    def plot_QUANTITY(
      output_path, # directory where figures need to be stored
      observational_data_path, # directory where observational data is found
      name_list, # list of labels that identify different simulations
      all_galaxies_list, # list of global galaxy data arrays
      prefix="", # prefix to add to image file names
    )
    ```
    The `prefix` argument is important if you also want to make the same plots for individual
    galaxies, as for example done with `plot_KS_relations`. This makes it possible to reuse the
    same plotting function for both the global and the individual plots, and guarantees that
    the latter use a name that is unique for a particular galaxy.
    Each plot should be added to a dictionary with the following general structure:
    ```
    plots = {
      SECTION_TITLE: {
        FIGURE_NAME.png: {
          "title": "TITLE",
          "caption": "CAPTION",
        }
      }
    }
    ```
    This information is used for the web page. The dictionary should be returned by the plotting
    function. Note that images that are not listed in this dictionary will not appear on the web
    page.
