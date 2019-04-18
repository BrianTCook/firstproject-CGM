# firstproject-CGM
A sampling of codes written specifically for Brian Cook's First Research Project as part of the Astronomy Research master's program at the Leiden Observatory.

The EAGLE team maintains catalogs of galaxies within simulations of different sizes and SPH particle resolutions in the form of SQL relational databases. The halos_snap=28_L0025N0376_README file is a sample SQL query used to find the listed attributes of all of the galaxies with the indicated "snapshot", i.e. redshift (e.g., snapshot = 28 -> redshift = 0). The galaxy IDs, stellar masses, star formation rates, etc. can then be stored in the form of .csv files.

Using halopicker.py we can then employ the pandas library to find galaxies with particular attributes. Depending on the quantity being considered it useful to be able to select out by value (e.g., a star formation rate >= 10^{-11} solar masses per year) or by percentile (e.g., galaxies with a stellar mass greater than the median). The script then saves the snapshot-dependent galaxy IDs (there is some stuff about merger trees in the EAGLE documentation that expands upon why a galaxy ID may change with time) in one .txt file and all of the relevant galaxy attributes in another. These .txt files will then be used to locate galaxies within the simulated region and inform future scripts how much of the cosmological box needs to be included to encapsulate the entire halo (galaxy + CGM).

There is an existing script (written by Nastasha Wijers) that reads in the EAGLE data for a particular region of the cosmological box, generates projections of various quantities like dark matter mass or Si IV column density, and stores them as .npz files. As elaborated on in section 2 of the thesis we often "slice" the cosmological box in such a way that the contamination from other objects along the line of sight is similar to that found in observations, so the outputsfromslices.py script executes relevant analyses in the following way:

1.) reads in all of the slice projections (as .npz files)

2.) iterates through each slice, finding all of the galaxies we are interested in analyzing that live within it

3.) determines which pixels of the entire slice projection can be attributed to the galaxy

4.) allocates each pixel of a galaxy projection into an appropriate impact parameter bin (e.g., a pixel at r = sqrt(2) R_{vir} would be in the 100th percentile impact parameter bin).

5.) finds statistics for all of the galaxies being analyzed at each impact parameter (e.g., what is the 75th percentile for H I       column density at r = 0.6 R_{vir} for star-forming galaxies with stellar mass 8.5 < log(M_{star}/M_{sun}) < 9?)

Another capability of outputsfromslices.py is computing the covering fraction, which determines which percentage of pixels at a particular impact parameter exceed an imposed threshold value (which is usually determined from observational constraints).
