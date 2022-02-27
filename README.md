# Tensor-Compress
## About
This is a small command line tool to experiment with tensors and video compression.
A video can be interpreted as a 4th-order tensor with dimensions 
```
color channel x width x height x frame
```
For matrices (or second order tensors), the best low rank approximation with respect to the Frobenius norm
is given by its [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition). Taking only the parts corresponding to the k largest singular values, one gets the best rank-r approximation. If the k is sufficiently small, the singular values together with the required basis vectors require less space than the initial matrix.

While the generalisation of the SVD to arbitary tensors in the form of the [higher-order singular value decomposition](https://en.wikipedia.org/wiki/Higher-order_singular_value_decomposition) does not have the same strong guarantee, it can be used in a similar fashion. The decomposition yields a number of matrices equal to the tensor order and a new tensor containing the singular values. By truncating this new tensor, one gets a quasi-optimal solution to the rank optimization problem.
## Build
Tensor-Compress requires a C++17 compiler and cmake (>=3.12).
Most dependencies are integrated as submodules and are thus taken care of by a recursive clone or a submodule update. However, ffmpeg needs to be available on the system. On windows, a dev lib can be aquired through [vcpkg](https://vcpkg.io/en/index.html). Once set up probably, run
```
$ git clone --recursive https://github.com/Thanduriel/tensorCompress
$ cd tensorCompress
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_TOOLCHAIN_FILE=<path/to/vcpkg.cmake>
```

## Usage
To get an overview of the available options try
```
tensorComp --help
```
As input file any video file that ffmpeg can handle is valid. For the output file, the ending should be `.avi` to store a lossless version of the video reconstructed from the truncated tensor. Other video formats are not supported. If the output file has the ending `.ten`, truncated tesor itself is stored instead. This option is of limited use currently, as the resulting file is quite large and the only operation that can be done with it, is to load it again to encode it as an `.avi` file.

Supported pixel formats for `--pix_fmt` are *YUV444* and *RGB*. The different color spaces effect the result, especially if the color dimension is truncated.

Truncation modes for `--trunc` are *rank*, *tolerance* and *tolerance_sum*. Together with the truncation threshold values, this rule determines how many singular values are kept in each dimension. Expected are 4 values, one for each dimension and in the order color width height frames. In case of *tolerance* and *tolerance_sum*, the values are interpreted as float threshold for singular values to keep, for *rank* it should be integers describing the size of the resulting tensor.


## Example
coming soon
as with pictures not really practical; poor basis leads to quickly perived artifacts