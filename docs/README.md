# CSC8101 2019-20 Coursework assignment

This coursework uses the [Movielens 20M](https://grouplens.org/datasets/movielens/20m/) dataset, specifically the ```ratings``` dataset which contains 20,000,263 records. Each record is a movie rating with the following schema:

```
 |-- userId: integer (nullable = true)
 |-- movieId: integer (nullable = true)
 |-- rating: double (nullable = true)
 |-- timestamp: integer (nullable = true)
```

There are 138,493 unique users, and 26,744 unique movies.
 
The dataset is available on an Azure Blob store. For convenience, here is Spark code to load the dataset:

```
import pandas as pd
ratingsURL = 'https://csc8101storageblob.blob.core.windows.net/datablobcsc8101/ratings.csv'
ratings = spark.createDataFrame(pd.read_csv(ratingsURL))
```
## Task 1 [5 marks]

Produce summary statistics from the ```ratings``` dataset:

- average number of ratings per users
- average number of ratings per movie
- histogram showing the distribution of movie ratings per user
- histogram showing the distribution of movie ratings per movie
(- others that you think may be informative, in view of the use of an algorithm that deals with sparsity in the user-item matrix)

(hint: explore the Databricks notebooks ```display()``` facility, [documented here](https://docs.databricks.com/notebooks/visualizations/index.html)

## Task 2: build a recommendation model to predict movie ratings from users [20 marks]

1. Using the full ```ratings``` dataset, train a recommender model using the ALS algorithm from the Spark MLlib library, which is [documented here](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.recommendation.ALS). Make sure you create separate training and test sets and measure your perfomenance on both,  using a RMSE performance metric.

2. using a GridSearch strategy, tune the ```rank``` and ```regParam``` hyperparameters of the model. Experiment with various values of the parameters and report on the time spent in GridSearch as a function of the size of the parameter space (this is the product of the possible values for each of the parameters).
Can you tune one parameter at a time? are their settings independent of each other?

Report on your best performance and comment on any model overfitting issues you may have spotted.

Hint: we are aware that a near-complete solution for part (1) is available from the Spark MLlib documentation:  [Spark doc for Collaborative Filtering](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html). However be aware that the input data format may be different.

## Task 3. [10 marks]

Produce a small-scale version of the ```ratings.csv``` file, by downsampling it so it includes 27,000 unique users, and _all ratings given by those users_. Note that you cannot simply randomly sample the ratings file, instead you need to create a set of users, sample 27,000, and then collect all of their ratings from the _ratings_ dataset.
Save this as a ```parquet```, a binary format that is much faster to load than csv.

Call this ```ratings-small.parquet```. This is the dataset you will use for the remainder of the couresework.

## Task 4: Generate a user-user network from the ratings dataset.  [15 marks]

The ratings dataset may be viewed as a user-movies matrix, where each cell is a rating (with a timestamp).
In this task you create a representation of a graph of users, where each user is represented by a node, and there is an edge between two users u1, u2 if u1 and u2 have rated the same movie. The *weight* of the edge is the number of the same movies that have both rated. Note that for simplicity here we ignore the values of the ratings. 

**Added 10/2/2020: consider thresholding the weights to reduce the overall connectivity**

The graph you will generate is likely to have 1 or max 2 connected components, i.e., include the entire graph.
this is because weights are being ignored.
To alleviate this, think of a criterion you can use to prune low-weight edges. This can be a fixed threshold, or one determined in the basis of the weights distribution. Make sure you end up with >2 significant connected components as a result of this additional operation, and be prepared to discuss your solution during the viva.

## Task 5: Discover the connected components of the graph, and select the largest for the next task.  [5 marks]

Use the GraphFrames API to calculate the connected components of the graph.
This is very simple to do as the ```connectedComponents()``` method will do all the work for you. 
The method returns a list of nodes with the number of the component (0,1,...) they belong to.

You now need to generate a representation of the subgraph containing the largest component, which you will use in task 6.
Note that the vertices and edges of a Graph are just dataframes, so they can each be saved and then retrieved as .parquet files.

Documentation:
[GraphFrame examples Databricks:](https://docs.databricks.com/spark/latest/graph-analysis/graphframes/index.html)
[GraphFrame Python API](https://graphframes.github.io/graphframes/docs/_site/api/python/index.html)

## Task 6: Implement the Girvan-Newman algorithm and apply it to the user graph you have created in Task 4.  [50 marks]

In this task you are required to analyse how the overall GN algorithm can be parallelised, at least in part, by distributing the computation over multiple workers using the MapReduce pattern.

The input to this task is the subgraph you produced in *Task 5*, which contains the largest connected component out of the entire graph. If you have saved the graph to file as dataframes, you will reload it in memory now.

_Hint_: begin by creating a representation of the graph as an adjacency list. **there is no need to use the GraphFrame representation of the graph**

also, for the purpose of calculating betweenness **you can assume all edges have equal weight.** 

You will then implement code to calculate the betweeness of each edge in the graph, being aware of which parts of your code will be executed on the driver (master), and which are run on workers. Correspondingly, you need to figure out which data can be sent to the workers and operated on in parallel, and how the final solution is collected on the driver.

The result should be a function that takes the input graph and returns a list of edges sorted in decreasing order of their betweenness.

A very small test graph is given below for you to experiment with. Note that this is the same graph used in the lecture notes.

Once you have working code, apply your method to the connected component graph that you produced in task 4.
Looking at the resulting betweeness values, remove the top-K edges and report on how many communities are generated. 

Report on the performance of your implementation and be prepared to discuss its scalability during the viva, considering your design choices for data distriubtion.

## Bonus task (for extra marks)
Apply your algorithm to the _full_ ```ratings``` dataset and discuss its performance and scalability properties.

**Important:** this implementation can be time-consuming and rather tricky. If you prefer, you may request to use a ready-made partial implementation (that is, the workers code) so all you have to do is figure out how to distribute data to the workers and combine the partial results on the driver. _50% of the marks for this task are deducted from your total marks if you choose this option._

### Sample graph for testing your GN implementation.

```
vertices = sqlContext.createDataFrame([
  ("a", "Alice", 34),
  ("b", "Bob", 36),
  ("c", "Charlie", 30),
  ("d", "David", 29),
  ("e", "Esther", 32),
  ("f", "Fanny", 36),
  ("g", "Gabby", 60)], ["id", "name", "age"])
```

```
edges = sqlContext.createDataFrame([
  ("a", "b", "friend"),
  ("b", "a", "friend"),
  ("a", "c", "friend"),  
  ("c", "a", "friend"),
  ("b", "c", "friend"),
  ("c", "b", "friend"),
  ("b", "d", "friend"),
  ("d", "b", "friend"),
  ("d", "e", "friend"),
  ("e", "d", "friend"),
  ("d", "g", "friend"),
  ("g", "d", "friend"),
  ("e", "f", "friend"),
  ("f", "e", "friend"),
  ("g", "f", "friend"),
  ("f", "g", "friend"),
  ("d", "f", "friend"),
  ("f", "d", "friend")
], ["src", "dst", "relationship"])
```
a GraphFrame representation of this graph is obtained simply as:
```
from graphframes import *
g = GraphFrame(vertices, edges)
```

Note; you will need to have installed the GraphFrame library on your cluster, by following these instructions:

from the Cluster Databricks UI, click on "libraries" -- install new --> Library source: Maven --> Coordinates / search packages "graphframes". select the package and confirm "install"


## References

1. Girvan M. and Newman M. E. J., Community structure in social and biological networks, Proc. Natl. Acad. Sci. USA 99, 7821–7826 (2002).
2. Freeman, L., A Set of Measures of Centrality Based on Betweenness, Sociometry 40, 35–41  (1977).
3. E. W. Dijkstra, A note on two problems in connexion with graphs. Numerische Mathematik, 1:269–
271, (1959)
4. GitHub - networkx/networkx: Official NetworkX source code repository, https://github.com/networkx/networkx
5. Takács, G. and Tikk, D. (2012). Alternating least squares for personalized ranking. 
Proceedings of the sixth ACM conference on Recommender systems - RecSys '12. 
[online] Available at: https://www.researchgate.net/publication/254464370_Alternating_least_squares_for_personalized_ranking
