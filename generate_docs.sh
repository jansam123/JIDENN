#!/bin/bash
# Generate documentation for jidenn
rm -rf docs
pdoc --html --force -o docs --config show_inherited_members=True --config  latex_math=True jidenn
mv docs/jidenn/* docs
rm -rf docs/jidenn
