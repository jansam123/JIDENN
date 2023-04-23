#!/bin/bash
# Generate documentation for jidenn
rm -rf docs
pdoc --html --force -o docs --template-dir docs_template jidenn
mv docs/jidenn/* docs
rm -rf docs/jidenn
