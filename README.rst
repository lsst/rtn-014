.. image:: https://img.shields.io/badge/rtn--014-lsst.io-brightgreen.svg
   :target: https://rtn-014.lsst.io
.. image:: https://github.com/lsst/rtn-014/workflows/CI/badge.svg
   :target: https://github.com/lsst/rtn-014/actions/

#############################################################
Lunar Complications in the Scheduling of Deep Drilling Fields
#############################################################

RTN-014
=======

The cadence of measurements of objects in in Legacy Survey of Space and Time (LSST) Deep Drilling Fields (DDFs) does not match the observing cadence for all objects, because objects detected in one sequence of exposures may be too faint to be detected in others: even when fields are observed at an optimum time within each night, the limiting magnitude can vary by more than 2 magnitudes over a lunation.
This note examines the effects of the variation in sky brightness due to the moon on the cadence of measurements of objects in Legacy Survey of Space and Time (LSST) Deep Drilling Fields (DDFs).
Plots of the variation in limiting magnitude by night are shown for each DDF, and physical explanations for its major characteristics discussed.
A few strategies for minimizing the impact are described, trade-offs highlighted, and a list of related questions on science requirements raised.

Links
=====

- Live drafts: https://rtn-014.lsst.io
- GitHub: https://github.com/lsst/rtn-014

Build
=====

This repository includes lsst-texmf_ as a Git submodule.
Clone this repository::

    git clone --recurse-submodules https://github.com/lsst/rtn-014

Compile the PDF::

    make

Clean built files::

    make clean

Updating acronyms
-----------------

A table of the technote's acronyms and their definitions are maintained in the ``acronyms.tex`` file, which is committed as part of this repository.
To update the acronyms table in ``acronyms.tex``::

    make acronyms.tex

*Note: this command requires that this repository was cloned as a submodule.*

The acronyms discovery code scans the LaTeX source for probable acronyms.
You can ensure that certain strings aren't treated as acronyms by adding them to the `skipacronyms.txt <./skipacronyms.txt>`_ file.

The lsst-texmf_ repository centrally maintains definitions for LSST acronyms.
You can also add new acronym definitions, or override the definitions of acronyms, by editing the `myacronyms.txt <./myacronyms.txt>`_ file.

Updating lsst-texmf
-------------------

`lsst-texmf`_ includes BibTeX files, the ``lsstdoc`` class file, and acronym definitions, among other essential tooling for LSST's LaTeX documentation projects.
To update to a newer version of `lsst-texmf`_, you can update the submodule in this repository::

   git submodule update --init --recursive

Commit, then push, the updated submodule.

.. _lsst-texmf: https://github.com/lsst/lsst-texmf
