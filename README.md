online_boosting
===============

A suite of boosting algorithms and weak learners for the online learning setting.

## Ensemblers

Implementations for the following online boosting algorithms are provided:

1. Online AdaBoost (OzaBoost), from [Oza & Russell](http://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf).
2. Online GradientBoost (OGBoost), from [Leistner et al.](http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=5457451).
3. Online SmoothBoost (OSBoost), from [Chen et al.](http://ntur.lib.ntu.edu.tw/retrieve/188503/07.pdf).
4. OSBoost with Expert Advice (EXPBoost), again from [Chen et al.](http://ntur.lib.ntu.edu.tw/retrieve/188503/07.pdf).
5. OSBoost with Online Convex Programming (OCPBoost), again from [Chen et al.](http://ntur.lib.ntu.edu.tw/retrieve/188503/07.pdf).

The corresponding Python modules can be found in the 'ensemblers' folder, named as above.

## Weak Learners

The package also includes implementations for a number of online weak learners, all of which can be plugged in to the above online boosting algorithms. Some of the key weak learners include:

1. Perceptrons.
2. Naive Bayes (Gaussian & Binary).
3. Random Decision Stumps.
4. Incremental Decision Trees, based on the [DTree](https://github.com/chrisspen/dtree) module.

## Dependencies

The ensemblers and weak learners are generally dependent on Numpy and Scipy. Some of the weak learners (in particular, those prefixed with "sk") are dependent on [scikit-learn](http://scikit-learn.org/stable/). File I/O is done through [YAML](http://en.wikipedia.org/wiki/YAML) using the [PyYAML](http://pyyaml.org) package.

A full list of dependencies is available in the [requirements.txt](https://github.com/crm416/online_boosting/blob/master/requirements.txt).

## License

MIT © [Charles Marsh](http://www.princeton.edu/~crmarsh)