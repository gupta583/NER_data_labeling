# NER_data_labeling
This model labels the data using BERT pretrained model.
Demo at http://quantanalytics.tech/
# Setting up python environment 

pip install -r requirements.txt

# Train model 
python model/train_model.py

#For inference
python app.py


# Introduction

Labelled data is crucial for any machine learning model. To get the machine learnig data companies uses Excel sheet to maintain the labels of data. This project has done with datasaur.ai for labeling data tool. This uses Bert Model to label data automatically. 

Bert Pretrained model for text classification is used and transfer learning has applied to learn new labels using Pytorch.
