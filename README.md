* make              : Creates the output folders
* make parallel     : Builds the parallel version
* make sequential   : builds the sequential version
* ./parallel        : Runs the parallel version and fills the 'data/' directory with stored time steps
* make plot         : converts saved time steps to png files under 'images/', using gnuplot. Runs faster if launched with e.g. 4 threads (make -j4 plot).
* make movie        : converts collection of png files under 'images' into an mp4 movie file, using ffmpeg
* make check        : builds both executeables and compares their output
