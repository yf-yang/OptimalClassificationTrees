# OptimalClassificationTrees
Python Implementation of [Optimal Classification Trees](https://dspace.mit.edu/handle/1721.1/110328).

MIT license. Contributions welcome!

### Installation
Please have cvxpy installed with there [installation guide](http://www.cvxpy.org/install/index.html).
A MIP solver should be installed. Please refer to the guide [here](https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver). Gurobi is recommended, now the repo only supports gurobi.

### Goals
-   Implement scikit-learn compatible APIs.
-   Implement it fast (at the expense of memory).
-   Implement APIs to interpret/visualize the model.

### TODO List
-   []  Validate performance on 53 [UCI ML datasets](http://archive.ics.uci.edu/ml/index.php).
-   []  Complete APIs as [DecisionTreeClassifier](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/tree.py) from sklearn.
-   []  Code refactor and add comments.
-   []  APIs to interpret the model.
-   []  A document to explain how it is implemented.
-   []  Further improvement and experiments.

### Status
Now the model only has a naive fit/predict method, which should be validated.