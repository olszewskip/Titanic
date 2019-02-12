# Titanic
The classic binary classification, the *Kagle*'s 'hello-world' (https://www.kaggle.com/c/titanic).

You can find my step-by-step notes here. The project is split into two standalone jupyter-notebooks. 

desc | link | notable
--- | --- | ---
Examine the data. Build a robust and reproducible preprocessing pipe. Visualize the data. | https://github.com/olszewskip/Titanic/blob/master/preprocess_visualize.ipynb | custom-Transformer-classes, ColumnTransformers, FeatureUnion, Pipeline; PCA, TSNE
Fit and tune classifiers using grid-search with cross-validation. Report accuracy, confusion-matrix, ROC-curve. Select best based on independent validation score. | https://github.com/olszewskip/Titanic/blob/master/classify.ipynb | GridSearchCV, LogisticRegression, NaiveBayes, GradientBoosting, AdaBoosting, VotingClassifier

plus a snippet which prepares *Kaggle*-submittable csv: https://github.com/olszewskip/Titanic/blob/master/prepare_submission.ipynb
