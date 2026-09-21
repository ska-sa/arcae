set -ex

brew update
# Homebrew's llvm is deliberately absent: CMake resolves cc/c++ to
# /usr/bin first, so every build here uses AppleClang regardless, and
# llvm has no bottle for macOS 14 or for x86_64 macOS, making it a
# ~2 hour source build on exactly the runners the wheels are built on.
brew install bison flex ninja python
brew reinstall gcc  # Need for gfortran
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
