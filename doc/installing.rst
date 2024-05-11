**********
Installing
**********

Eelbrain can be installed from `conda-forge <https://conda-forge.org>`_ (pre-compiled) and from the Python Package Index (`PyPI <https://pypi.org/project/eelbrain/>`_; requires compilation).

The recommended way to use Eelbrain is in a separate `Conda environment <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_.
The recommended tool for managing Conda environments is `Mamba <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_.

A straight forward way to create an environment with the libraries you need is using an ``environment.yml`` file. The `Alice <https://github.com/Eelbrain/Alice>`_ repository's `environment.yml <https://raw.githubusercontent.com/Eelbrain/Alice/main/environment.yml>`_ can serve as a starting point)::

    $ mamba env create --file=environment.yml


In an existing environment, Eelbrain can generally be updated with the following command (assuming the target environment is currently active)::

    (eelbrain) $ mamba update eelbrain


Sometimes Mamba may run into difficulties while updating and it may be easier to create a new environment instead.

.. SEEALSO::
   For other methods for installing Eelbrain, see this `wiki page <https://github.com/christianbrodbeck/Eelbrain/wiki/Installing>`_.
