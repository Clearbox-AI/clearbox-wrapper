# FAQ

## Setup

**What should I do if I want to use a different ML model or dataset?**

In this case you should remove all files present in the _resources_ folder. If you only want to change your model, you can keep the _dataset.joblib_ file.

**Why is the setup taking so much time?**

The time required by the setup depends on the size of your dataset. During this operation our tool is training its own model to give you faster and more precise explanations. Feel free to enjoy a ☕️ in the meantime!



## Usage

**How to edit the port used by the application?**

By default the ClearBox App host is on port _8080_, while the MongoDB host is on port _27017_. You can change these default values, according to your needs, by modifying the _.env_ file in the root directory. 

**What does anomaly score mean?**

The anomaly score can be used to detect outliers, i.e. how much the input you provided differs from the ones you have used to train your ML model. You can find more details [here](./api/predictions.md).

**What does trust score mean?**

The trust score can be used to find how much you can trust your model for a certain prediction, i.e. when your ML model has troubles to distinguish between the predicted label and another class very close to the predicted one. It is based on a proxy metric presented in [this paper](https://arxiv.org/abs/1805.11783).


**What is an explanation by examples?**

This kind of explanation returns a series of instances, that are considered by the model very similar to the current query. These instances are taken from the original dataset that you used to train the ML model. The examples are divided into two sets: _pertinent examples_ and _not pertinent examples_. The former are examples that belong to the same class as your input instance, while the latter are classified by the model as a different class. 
This kind of explanation allows the user to understand which training points contributed most to the current model prediction.


**What is an explanation by local tree?**

An explanation by local tree creates a local surrogate in the form of a decision tree classifier. This surrogate can be used to have a simplified view on how your model behaves when in presence of small changes of the input features. It can be used to find anchoring rules or counter factual examples.


**Why do I need the feedback endpoint?**

Continuous monitoring and uncertainty analysis require the knowledge of what the model gets right or wrong over time. This information needs of course to be provided by the user. Feedback will be also used in the future to provide an active learning mechanism, in order to re-calibrate your ML model after a certain amount of time.


## Issues

**The application does not start because some resources are missing. What should I do?**

This means that you do not have all the required resources. In this case you can copy a set of resources from the ones that we provide as a demo, or you can create your own resources following the Jupyter Notebook included as part of our documentation.