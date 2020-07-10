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

Labelled data is crucial for any machine learning model. To get the machine learnig data companies uses Excel sheet to maintain the labels of data. This project has done with datasaur.ai for labeling data tool. datasaur.ai develops a software to label the data efficiently. 

For NER model, here I have used the BERT pretrained model for text classification.  To learn new labels Transfer Learnig has applied. Here I have loaded the weight of pretrained model then a new layer has been added for classification. The weight are update by calculating the gradient of binary cross entropy loss. I have used Linear Schedueler to make learning faster compared to constant linear rate. Figure shows the systematic diagram for transfer learning. 

![picture](TL.png | height="50")

Bert Pretrained model for text classification is used and transfer learning has applied to learn new labels using Pytorch.
