#! /usr/bin/env bash

help()
{
    echo
    echo "Plot 2D Wave Equation"
    echo
    echo "Syntax"
    echo "--------------------------------------------------------"
    echo "./plot_results.sh [-m|n|h] [data_folder]                "
    echo
    echo "Option    Description      Arguments   Default"
    echo "--------------------------------------------------------"
    echo "m         y size           Optional    128    "
    echo "n         x size           Optional    128    "
    echo "h         Help             None               "
    echo
    echo "Example"
    echo "--------------------------------------------------------"
    echo "./plot_solution.sh -m 128 -n 128"
    echo
}

#-----------------------------------------------------------------
set -e

M=128
N=128

# Check if the data folder is provided
if [ $# -lt 1 ]; then
    echo "Error: No data folder provided."
    help
    exit 1
fi

# Parse options and arguments
while getopts ":m:n:h" opt; do
    case $opt in
        m)
            M=$OPTARG;;
        n)
            N=$OPTARG;;
        h)
            help
            exit;;
        \?)
            echo "Invalid option"
            help
            exit;;
    esac
done

# Shift parsed options so that the remaining arguments start at $1
shift $((OPTIND - 1))

# Ensure that the data folder is provided and exists
DATAFOLDER=./data
if [ ! -d "$DATAFOLDER" ]; then
    echo "Error: Data folder $DATAFOLDER does not exist."
    exit 1
fi

#-----------------------------------------------------------------
# Set up the size of the grid based on the options passed
SIZE_M=`echo $M | bc`
SIZE_N=`echo $N | bc`

# Ensure the output directory exists
mkdir -p images

# Loop through all .dat files in the data folder
for DATAFILE in "$DATAFOLDER"/*.dat; do
    # Skip if no .dat files are found
    if [ ! -f "$DATAFILE" ]; then
        echo "No .dat files found in the folder."
        exit 1
    fi

    # Create the corresponding output image file name
    IMAGEFILE=`echo $DATAFILE | sed 's/dat$/png/' | sed 's/data/images/'`

    # Run the gnuplot command to create the plot in the background
    (
        cat <<END_OF_SCRIPT | gnuplot -
        set term png
        set output "$IMAGEFILE"
        set zrange[-1:1]
        splot "$DATAFILE" binary array=${SIZE_M}x${SIZE_N} format='%double' with pm3d
END_OF_SCRIPT

        echo "Plot saved to $IMAGEFILE"
    ) &   # Run in the background

done

# Wait for all background processes to finish
wait

echo "All plots have been generated."
