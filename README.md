

Answer on the ipthon doc attached.

Classifier which can’t be used for the task
a)Linear SVM 
It currently only supports binary classification. This cannot be used for for multi-class classification.

Classifier which can be used:
b) Decision Tree Classifier
A decision tree is a flow-chart-like structure, where each internal (non-leaf) node denotes a test on an attribute, each branch represents the outcome of a test, and each leaf (or terminal) node holds a class label. The topmost node in a tree is the root node. 

model = graphlab.decision_tree_classifier.create(train_data, target=’X1’, max_iterations=2, max_depth = 3)

predictions = model.predict(test_data)
We see good accuracy results because the decision tree model is very good at handling tabular data with numerical features, or categorical features with fewer than hundreds of categories.

c) Nearest Neighbor Classifier :
The nearest neighbors classifier predicts the class of a data point to be the most common class among that point's neighbors.
we create the model and generate predictions:
m = graphlab.nearest_neighbor_classifier.create(train_data, target=’X1’,features=[‘bow’])
predictions = m.classify(test_data, max_neighbors=20, radius=None).
62% accuracy seems low, but remember that we are in a multi-class classiciation setting. Adding the text feature appears to slightly improve the accuracy of our classifier to 64%.
d) Boosted Trees Classifier
The Gradient Boosted Regression Trees (GBRT) model (also called Gradient Boosted Machine or GBM) is a type of additive model that makes predictions by combining decisions from a sequence of base models.
model3 = graphlab.boosted_trees_classifier.create(train_data, target='X1',features = ['bow','tfidf'])
 predictions = model3.predict(test_data)
boosted trees model is very good at handling tabular data with numerical features, or categorical features with fewer than hundreds of categories
I got a significant performance improvement when I converted the bow vector into tf-idf in my experiments and My classifier tends to perform even better when we consider both TFIDF and BOW as features as I record better Recall accuracy precision and auc.

2. precision is "how useful the search results are", and recall is "how complete the results are" i.e., precision is the fraction of retrieved documents that are relevant to the query and Recall in information retrieval is the fraction of the documents that are relevant to the query that are successfully retrieved. Better precision, recall and Accuracy, better the model is

Markup : ![picture alt](https://github.com/nandhiniramanan5/Graphlab-for-multi-class-text-classification/blob/master/Capture.JPG "Accuracy table")


It is always recommended to try different methods and see which one performs better on the given data. Using GraphLab Create it is easy to interchange methods and find the right one for my needs
My Decision Tree Classifier performs the best with an accuracy, Precision, Recall and AUC values.
