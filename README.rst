Active labeling: Streaming stochastic gradients
===============================================
:Topic: Implementation of the NeuRIPS 2022 paper [CAB22]_.
:Author: Vivien Cabannes
:Version: 0.0.1 of 2022/10/10

Notes
-----
Currently, the library has not been formated to be used easily out of the box.
Most of the interested code is indeed in the script folder.
Eventually, an effort to make it scikit-learn friendly should be consider.
Until then, one can reuse the code they are interested in, or implement it from scratch following the paper explications.

The code base relies on the stochastic gradient strategy developed for least-square and absolute-deviation regression or surrogate. 
If similar derivations exists for other losses (such as logistic or hinge) that would lead to easy to implement algorithm is an open question.


Installation from source
------------------------
Download source code at https://github.com/VivienCabannes/active_labeling/archive/master.zip.
Once downloaded, install it with

.. code:: shell
   $ pip install .

References
----------
.. [CAB22] Active labeling: Streaming Stochastic Gradients, Cabannes et al., *NeurIPS*, 2022.
