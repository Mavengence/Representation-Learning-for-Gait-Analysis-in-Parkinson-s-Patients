<div style="background-color:white">
  <div align="center">
    <img src="./imgs/techfak_logo.jpg" width="700" height="250">
    <hr>
    <h1 style="color:black">
      Representation Learning For Gait Analysis in Parkinson’s Patients
    <h1>
    <h3 style="color:black">Tim Löhr and Christoph Popp<h3>
    <img src="./imgs/madi_logo.png" width="400">
  </div>
  <hr>
</div>

## Abstract
This project aims to quantify how accurately Morbus Parkinson's can be classified by different types of deep learning architecture without preprocessing the original sensor data. For this purpose, four different architectures (LSTM, ResNet, a basic autoencoder and a ResNet autoencoder) were used to evaluate the accuracy. The data was collected from patients at the University Hospital of Erlangen. Different severity levels of Parkinson's were regarded as being deceased. In this regard, this project performed a binary classification task (healthy and deceased). It shows, that a ResNet autoencoder predicts Parkinson with 87% accuracy and can be used as a decision support system for doctors.


### Structure

```
+-- Code
|   +-- Notebooks                        
|   |    +-- Introduction.ipynb
|   |    +-- Prediction.ipynb
|   |    +-- ResNet.ipynb
|   +-- Models                        
|   |    +-- lstm
|   |    +-- autoencoder_loss
|   |    +-- autoencoder_sampler
|   |    +-- resnet
+-- Report
|   +-- Final Paper
|   +-- Related Work Paper
|   +-- Bibliography.bib
|   
+-- imgs                    
+-- requirements.txt                    
+-- README.md
+-- .gitignore              
```

## Links to Ressources

- Introduction Notebook as [iPython](https://gitlab.cs.fau.de/uj10yjun/representation-learning-for-gait-analysis-in-parkinson-s-patients/-/blob/master/notebooks/Introduction.ipynb)
- Prediction Notebook as [iPython](https://gitlab.cs.fau.de/uj10yjun/representation-learning-for-gait-analysis-in-parkinson-s-patients/-/blob/master/notebooks/Prediction.ipynb)
- Final Paper as [PDF](https://gitlab.cs.fau.de/uj10yjun/representation-learning-for-gait-analysis-in-parkinson-s-patients/-/blob/master/reports/MLTS_Tim_Loehr_Christoph_Popp.pdf)


## Train Parameters

```
  --model_name [lstm, resnet, autoencoder_sampler, autoencoder_loss]
  --max_batch_len [int between 1 and 13000]
  --batch_size [int between 1 and 200]
  --learning_rate [float between 1 and 0.00001]
  --epochs [int between 1 and infinite]
```

## Example Train Command

```
  python --model_name resnet --learning_rate 0.001 --epochs 50 max_batch_len 4096
```

## Ressources

- PyTorch Lightning: https://pytorch-lightning.readthedocs.io/en/stable/introduction_guide.html#predicting
- Google Colab for training: https://colab.research.google.com/


### Prerequisites

```
The dependencies to this project are stored in the file:
   - requirements.txt

We use python version 3.7.4
```

## Authors

* **Tim Löhr** - If you have questions you can contact me under timloehr@icloud.com
* **Christop Popp** - If you have questions you can contact me under christoph.popp@fau.de


## License

This project was done for the project in Machine Learning Timeseries from the Machine Learning and Data Analytics Lab at the Friedrich Alexander University in Erlangen-Nürnberg.

## Acknowledgments

* Thanks a lot to Falk Pulsmeyer from the Machine Learning and Data Analytics Lab for a really good supervising through all my project. I can totally recommend this seminar!
