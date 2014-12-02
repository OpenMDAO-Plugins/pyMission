=============
Installation
=============

This plugin requires a number of outside libraries. You'll need to aquire pyopt_sparse from the University of Michigan MDO lab. 
You'll also need the `MBI <http:\\https://github.com/hwangjt/MBI>` library, which provides a multivariate b-spline interpolant. 
Lastly, you'll need to install the PyOpt_Sparse driver from the `pyoptsparse_driver` folder in this repository. 


To work with this plugin, clone the repo to your local machine. Then, in an activated openMDAO environment: 

::

    python setup.py develop


The source code for all the components and assemblies is in the `src\pyMission` directory. All the different use cases are in 
the `sample_cases` directory. the sample cases include plotting scripts to help visualize the results. 





