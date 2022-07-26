# This script will clean all the built files, and rebuilt the project from zero.
# It can directly run from "script" directory.
# This script is not recommend to use, since it is too slow.

# Build project CudaKernel
echo -e "\n Building CudaKernel... \n"
cd cudaLib/build
./cleanBuild.sh
echo -e "Build files cleaned. \n"
cmake ..
echo -e "Cmake CudaKernel done. \n"
make
echo -e "CudaKernel built. \n"

# Build project LBM
echo "\n Building LBM \n"
cd ../../build
./cleanBuild.sh
./cleanVtkData.sh
echo -e "Build files & vtk data cleaned. \n"
cmake ..
echo -e "Cmake CudaKernel done. \n"
make
echo -e "CudaKernel built. \n"
