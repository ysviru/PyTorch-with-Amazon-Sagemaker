# Using Amazon Sagemaker with PyTorch

## 1. A simple example of Hyperparameter tuning with Sagemaker. 
   
   This is demonstrated in hyperparameter_tuning_simple_tutorial/pytorch_sagemaker_tutorial.ipynb. This is based on Amazon's original tutorial here -- https://github.com/awslabs/amazon-sagemaker-examples/tree/master/hyperparameter_tuning/pytorch_mnist. This tutorial itself is quite good, but I sought to make a simpler one to --- a) Make sure I understood how HyperparameterTuner works, and b) To help clarify some things that maybe useful to folks new to Hyperparameter tuning with Sagemaker. 
   
   ### How is this different from Amazon's tutorial? 
   
   I wanted to create the most minimal example demonstrating how to tune PyTorch model with Amazon Sagemaker. So I made the following changes to Amazon's original tutorial: 
   1. Instead of using MNIST, I used a synthetic dataset created by sklearn.datasets.make_classification. This simplifies data loading etc.  
   2. Instead of using a conv net, I am using a simple feedforward network. 
   3. I DO NOT include the code to perform distributed training (i.e., using torch.nn.DataParallel, torch.distributed, or torch.utils.data.distributed) to make the code simpler. 
   4. Finally, I stressed the point about the need to use logger so as to make HyperparameterTuner perform the tuning correctly. Although, Amazon's tutorial is quite good as it is, I wish they stressed this point more. 
   
   ### Steps to perform tuning in Jupyter notebook 
   1) Define synthetic data with sklearn.datasets.make_classification function. 
   2) Define a sagemaker session to obtain the default S3 bucket name and to get execution role. 
   3) Use the sagemaker session to upload data defined in a) on to S3. 
   4) Define a PyTorch model wrapper instance (defined as sagemaker.pytorch.PyTorch). This is an estimator instance that can perform model training. We need to provide the python script path to the constructor of sagemaker.pytorch.PyTorch to tell it the location of our custom PyTorch model and our custom methods to perform dataloading, training and testing. 
   5) Define a HyperparameterTuner instance that wraps the estimator (sagemaker.pytorch.PyTorch) instance. 
   6) Use the HyperparameterTuner instance we perform model training. 
   
   ### Note about entry point script that defines the custom PyTorch model
   In point 4) above, the python script defines the following: 
   1) logger instance for logging metric information such that the logged metric values can be used by HyperparameterTuner to decide which hyperparameter setting is better.  
   2) PyTorch model defined with class Net(). This is a simple feedforward network with two layers initialized with Xavier Glorot and Yoshua Bengio 2010 method of weight initialization.   
   3) Define two functions to load training and validation datasets. 
   4) Define train method containing the usual training loop. 
   5) Define test and save_model methods.  
   6) It is important to define the name == "__main__" block in order to inform sagemaker.pytorch.PyTorch class about the entry point to the script. After parsing all requisite arguments (hyperparameters to the model and Sagemaker environment variables), we call the train() method. 
