Customer Churn Prediction for a telecommunication industry 
----------------------------------------------------------

An example of how to apply MLOps concepts like DVC, MLflow, Github actions for CI in machine learning

The repo shows basic concepts, such as:
--------------------------------------

DVC for data versioning
Scikit-learn pipelines to preprocessing and modeling.
Mlflow for tracking experiments
HyperOpt for faster Hyperparameter tuning
Docker for containarization of the web app
Github actions for continuous integration of changed Docker Image to Docker Hub

Common workflows:
-----------------

1] Versioning data

   * dvc init
   
   * dvc remote add --> added a local registry
   
   * dvc add --> the data you want to track
   
   * dvc push --> to local registry with a tag (so whenever we need a version of data, we can use these tags to dvc pull it)

2] Used Jupyter notebook for EDA and for model selection 

   * created a Column Transformer for transforming different columns
   
   * created a pipeline to pass this transformed columns with model
   
   * used imblearn for ImbPipeline as I wanted to use SMOTE for up sampling

3] Created a function which takes a model, preprocessor and calculates metrics.

4] Used Hyper opt for a algorithm which gave good result on base dataset

5] Used MlFlow for tracking metrics, parameters and model's pickle file for model registry

6] Created a train.py and parameters yaml file for training, tracking.

Docker:
-------

Created a web application using flask and dockerize, created a image

To use --> pull the image, used 5000 port but you can map it to any other port.

           docker run --name give_name -p your_port:5000 image_name
           
https://hub.docker.com/repository/docker/learningp/customer-churn-prediction/general

Github actions:
---------------

Created jobs to push images to docker hub registry, used Github's secrets for username and password.

Obtained username and password from Docker Hub

Further work:
-------------
To improve models metrics such as precision and recall, try to use other CI tools and also implement deployment of the application.
