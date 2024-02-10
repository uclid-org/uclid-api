=====
UCLID5 Python API
=====

The official repository for the UCLID5 Python API.


Installation
============
To install the UCLID5 Python API, you can use pip::

    pip install . # from the root directory of the project

This will install the `uclid` package and its dependencies.

Usage
=====
See tests for examples of how to use the API.


Making Changes & Contributing
=============================

See `CONTRIBUTING.rst` for more information on how to contribute to this project.

To run all tests and get a covereage report, just execute `tox` in the root
directory of the project.

Note that this project uses `pre-commit`, please make sure to install it before
making any changes::

    pip install pre-commit
    cd uclid-api
    pre-commit install

It is a good idea to update the hooks to the latest version::

    pre-commit autoupdate
