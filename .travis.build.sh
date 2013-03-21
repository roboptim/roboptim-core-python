#!/bin/sh

# Directories.
root_dir=`pwd`
build_dir=$root_dir/build
core_dir=$build_dir/roboptim-core

# Create layout.
mkdir -p $build_dir

# Checkout roboptim-core
cd $build_dir
git clone --recursive git://github.com/roboptim/roboptim-core.git
cd $core_dir
cmake . -DCMAKE_INSTALL_PREFIX=$install_dir
make install

# Build package
cd $build_dir
cmake $root_dir -DCMAKE_INSTALL_PREFIX=$install_dir
make
make test
