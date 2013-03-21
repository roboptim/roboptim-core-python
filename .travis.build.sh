#!/bin/sh

# Directories.
root_dir=`pwd`
build_dir="$root_dir/_travis/build"
install_dir="$root_dir/_travis/install"
core_dir="$build_dir/roboptim-core"

# Shortcuts.
git_clone="git clone --quiet --recursive --depth 1"

# Create layout.
rm -rf "$build_dir"
mkdir -p "$build_dir"

# Setup environment variables.
export PKG_CONFIG_PATH="$install_dir/lib/pkgconfig:$PKG_CONFIG_PATH"
echo $PKG_CONFIG_PATH

# Checkout roboptim-core
echo "Installing dependencies..."
cd "$build_dir"
$git_clone "git://github.com/roboptim/roboptim-core.git"
cd "$core_dir"
cmake . -DCMAKE_INSTALL_PREFIX:STRING="$install_dir"
make install

# Build package
echo "Building package..."
cd "$build_dir"
cmake "$root_dir" -DCMAKE_INSTALL_PREFIX="$install_dir"
make
make test
