# Advanced Usage

### Prerequisites

If you want to deploy your own machine learning model with the Clearbox APP you need to:
* Have the tabular dataset used to train the model saved as _.csv_
* Have your machine learning model serialized and saved on disk. As now the ClearBox APP was tested only with the following frameworks: _xgboost_, _sklearn_, _Keras_. 

### Resources setup

Before you can build the Docker image of the application and use it, you need to configure your resources (i.e. your dataset and your model) within the project. 
The first step is to copy both your dataset in _.csv_ format and your serialized model, inside the folder `setup/resources`. At this point it is possible to build the first Docker image that will allow you to convert your resources to make them usable by ClearBox App. This procedure must be performed the first time you run the application and whenever you make changes to your dataset or to your model. To start this process it is necessary, from your terminal, to move to the `setup` folder and then to run the following commands:
```sh
docker build --tag clearbox_setup .

docker run -p 8844:8888 --mount type=bind,source=$(pwd),target=/setup --mount type=bind,source=$(dirname $(pwd))/clearbox_backend/resources,target=/setup/resources clearbox_setup
```
Now you can open the URL [127.0.0.1:8844](http://127.0.0.1:8844) on your web browser, where you will find a Jupyter Notebook to follow with all the instructions to convert your resources using ClearBox AI's Library.

### Usage

After you set up your own resources you can start to use the application as described in the setup section.