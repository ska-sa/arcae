set -ex

brew update
# Homebrew's llvm is deliberately absent: CMake resolves cc/c++ to
# /usr/bin first, so every build here uses AppleClang regardless, and
# llvm has no bottle for macOS 14 or for x86_64 macOS, making it a
# ~2 hour source build on exactly the runners the wheels are built on.
brew install bison flex ninja python
# gfortran is needed by casacore's casa_scimath_f, both when vcpkg builds
# casacore from source on a binary cache miss and when linking the C++ test
# executables. Homebrew's unversioned gcc (16.x) lost its x86_64 macOS bottle
# in September 2026, making it a ~2 hour source build on the Intel runner;
# gcc@15 is still bottled for every runner used here. It installs the driver
# as gfortran-15, but vcpkg and CMake's check_language(Fortran) both look for
# plain gfortran, so expose it under that name.
brew install gcc@15
ln -sf "$(brew --prefix gcc@15)/bin/gfortran-15" "$(brew --prefix)/bin/gfortran"
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
