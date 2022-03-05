# Tensor-Compress
## About
This is a small command line tool to experiment with tensors and video compression.
A video can be interpreted as a 4th-order tensor with dimensions 
```
color channel x width x height x frame
```
For matrices (or second order tensors), the best low rank approximation with respect to the Frobenius norm
is given by its [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition). Taking only the parts corresponding to the k largest singular values, one gets the best rank-k approximation. If the k is sufficiently small, the singular values together with the required basis vectors require less space to store than the initial matrix.

A generalization of the SVD to arbitrary tensors is the [higher-order singular value decomposition](https://en.wikipedia.org/wiki/Higher-order_singular_value_decomposition). While it does not have the same optimality guarantee, it can be used in a similar fashion. The decomposition yields a number of matrices equal to the tensor order and a new tensor of the same order containing the singular values. By truncating this new tensor, one gets a quasi-optimal solution to the rank optimization problem.
## Build
Tensor-Compress requires a C++17 compiler (tested with msvc-17.1, gcc-10.3) and cmake (>=3.12).
Most dependencies are integrated as submodules and are thus taken care of by a recursive clone or a submodule update. However, ffmpeg needs to be available on the system.
In particular, the components `libavformat-dev`,`libavcodec-dev` and `libswscale-dev` should be installed.
On windows, a dev lib including all these components can be acquired through [vcpkg](https://vcpkg.io/en/index.html). Once set up probably, run
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
The example video shown here is Video32_Country_panorama from the [ITEC Short Casual Video Dataset](http://ftp.itec.aau.at/datasets/short-casual-videos/) (license [CC-4](https://creativecommons.org/licenses/by-nc-sa/4.0/)) with resolution 640x360 and 500 frames.

https://user-images.githubusercontent.com/5833421/155897158-1934cda5-dd4f-472d-883e-096f1de4392e.mp4

For easy viewing pleasure, the following approximated videos are also encoded as h264 but with a higher bit-rate.

Unfortunatly, even a minor reduction to rank (3x512x288x400) already results in some prominent artifacts, showing that this representation is unsuited for video compression.

https://user-images.githubusercontent.com/5833421/155897186-eb06ee06-9ab7-4c74-b584-b60b67d58e6b.mp4

Reducing the video to rank (1x32x28x25) the axis aligned segments which form the basis become visible in the spatial dimension. Furthermore, the adaptive frame-rate resulting from the truncation of the time dimension can be seen.

https://user-images.githubusercontent.com/5833421/155897578-8af870ab-d25a-4cd6-ae4f-732a20d39282.mp4
