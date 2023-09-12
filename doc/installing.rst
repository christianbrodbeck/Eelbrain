**********
Installing
**********

The recommended way to use Eelbrain is in a separate `Conda environment <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_.
First, install the `Anacond Python distribution <https://www.anaconda.com/download>`_.
Then, create an environment file with the libraries you need (for example, use this `environment.yml <https://raw.githubusercontent.com/Eelbrain/Alice/main/environment.yml>`_ as a starting point), and create the environment::

    $ conda env create --file=environment.yml


Eelbrain in an existing environment can generally be updated with the following command (assuming the target environment is currently active)::

    (eelbrain) $ conda update -c conda-forge eelbrain


Sometimes Conda will run into difficulties while updating and it may be easier to create a new environment instead.

.. TIP::
   If you often create or update environments, use `Mamba <https://github.com/mamba-org/mamba#readme>`_ as replacement for Conda.

For other ways of installing Eelbrain, see this `wiki page <https://github.com/christianbrodbeck/Eelbrain/wiki/Installing>`_.
