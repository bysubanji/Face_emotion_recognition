# Face-Recognition
# Face-Recognition
### Live Class Monitoring System(Face Emotion Recognition)
## Introduction
Emotion recognition is the process of identifying human emotion. People vary widely in their accuracy at recognizing the emotions of others. Use of technology to help people with emotion recognition is a relatively nascent research area. Generally, the technology works best if it uses multiple modalities in context. To date, the most work has been conducted on automating the recognition of facial expressions from video, spoken expressions from audio, written expressions from text, and physiology as measured by wearables.

Facial expressions are a form of nonverbal communication. Various studies have been done for the classification of these facial expressions. There is strong evidence for the universal facial expressions of seven emotions which include: neutral happy, sadness, anger, disgust, fear, and surprise. So it is very important to detect these emotions on the face as it has wide applications in the field of Computer Vision and Artificial Intelligence. These fields are researching on the facial emotions to get the sentiments of the humans automatically.


## Problem Statement
The aim of the project is to create a Facial Emotion Recognition System (FERS) that can detect students' emotional states in e-learning systems that use video conferencing.
This technology instantly conveys the emotional states of the students to the educator in order to create a more engaged educational environment.
Our results supported those of other studies that have shown that in e-learning systems, it is possible to observe the motivation level of both the individual and the virtual classroom.


Deep learning backed system not only solves the surveillance issue, but it also removes the human bias from the system, and all information is no longer in the teacher’s brain rather translated in numbers that can be analyzed and tracked.

I will solve the above-mentioned challenge by applying deep learning algorithms to live video data. The solution to this problem is by recognizing facial emotions.

## Dataset Information
I have built a deep learning model which detects the real time emotions of students through a webcam so that teachers can understand if students are able to grasp the topic according to students' expressions or emotions and then deploy the model. The model is trained on the FER-2013 dataset .This dataset consists of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised. Here is the dataset link:- https://www.kaggle.com/msambare/fer2013

## Dependencies
# Python 3
# Pandas and Numpy
# tensorflow
# Keras
# Pillow
# Streamlit
# Streamlit-Webrtc
# OpenCV

## Model Creation


## 1)Mobilenet:
MobileNet is an efficient and portable CNN architecture that is used in real-world applications. MobileNets primarily use depth-separable convolutions in place of the standard convolutions used in earlier architectures to build lighter models. MobileNets introduces two new global hyperparameters (width multiplier and resolution multiplier) that enable model developers to trade off latency or accuracy for speed and low size based on their needs.
 

 
## 2) Dexpression:
The suggested architecture outperforms the current state of the art utilizing CNNs by 99.6 percent for CKP and 98.63 percent for MMI. Face recognition software has a wide range of applications, including human-computer interface and safety systems. This is because nonverbal cues are vital types of communication that play an important part in interpersonal interactions. The usefulness and dependability of the suggested work for real-world applications is supported by the performance of the proposed architecture.
 

 
## 3) CNN:
Basic CNN architecture details:
• Input layer - The input layer in CNN should contain image data.
• Convo layer - The convo layer is sometimes called the feature extractor layer because features of the image are get extracted within this layer 
• Pooling layer - Pooling is used to reduce the dimensionality of each feature while retaining the most important information. It is used between two convolution layers.
• Fully CL - Fully connected layer involves weights, biases, and neurons. It connects neurons in one layer to neurons in another layer. It is used to classify images between different categories by training and placed before the output layer
• Output Layer - The output layer contains the label which is in the form of a one-hot encoded.
A Convolutional Neural Network (CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. 
 

## 4) Densenet
DenseNet was developed specifically to improve the declining accuracy caused by the vanishing gradient in high-level neural networks. In simpler terms, due to the longer path between the input layer and the output layer, the information vanishes before reaching its destination. 

 
## 5) ResNet:
The term micro-architecture refers to the set of “building blocks” used to construct the network. A collection of micro-architecture building blocks (along with your standard CONV, POOL, etc. layers) leads to the macro-architecture (i.e., the end network itself). First introduced by He et al. in their 2015 paper, the ResNet architecture has become a seminal work, demonstrating that extremely deep networks can be trained using standard SGD (and a reasonable initialization function) through the use of residual modules:

Further accuracy can be obtained by updating the residual module to use identity mappings, as demonstrated in their 2016 follow-up publication,
 Identity Mappings in Deep Residual Networks:

That said, keep in mind that the ResNet50 (as in 50 weight layers) implementation in the Keras core is based on the former 2015 paper.
Even though ResNet is much deeper than VGG16 and VGG19, the model size is actually substantially smaller due to the usage of global average pooling rather than fully-connected layers — this reduces the model size down to 102MB for ResNet50.


# 5) Model performance:

i) Confusion Matrix-
The confusion matrix is a table that summarizes how successful the classification model is at predicting examples belonging to various classes. One axis of the confusion matrix is the label that the model predicted, and the other axis is the actual label.

Precision, Recall, F1 score and Support-
Precision is the ratio of correct positive predictions to the overall number of positive predictions: TP/TP+FP
Recall is the ratio of correct positive predictions to the overall number of positive examples in the set: TP/FN+TP
It is also called the F Score or the F Measure. Put another way, the F1 score conveys the balance between the precision and the recall The F1 Score is the 2*((precision*recall)/(precision+recall))
Support is the number of actual occurrences of the class in the specified dataset. Imbalanced support in the training data may indicate structural weaknesses in the reported scores of the classifier and could indicate the need for stratified sampling or rebalancing

ii) Accuracy and loss curve-

Accuracy is a method for measuring a classification model's performance. It is typically expressed as a percentage. ... Accuracy is often graphed and monitored during the training phase though the value is often associated with the overall or final model accuracy. Accuracy is easier to interpret than loss.
Loss value implies how poorly or well a model behaves after each iteration of optimization. An accuracy metric is used to measure the algorithm's performance in an interpretable way. It is the measure of how accurate your model's prediction is compared to the true data.

# 6) Model Deployment-

### Creating Web App Using Streamlit-
Streamlit is a Python framework for developing machine learning and data science web apps that is open-source. Using Streamlit, we can quickly create web apps and deploy them. You can use Streamlit to make an app the same way you'd make a Python programme. It's possible with Streamlit. Working on the interactive loop of coding and viewing results is a pleasure. In the web application.

Deployment in cloud platform-
AWS (Amazon Web Services) is a comprehensive, evolving cloud computing platform provided by Amazon that includes a mixture of infrastructure as a service (IaaS), platform as a service (PaaS), and packaged software as a service (SaaS) offerings.



# Conclusion:

We began this project with two goals in mind: to achieve the maximum level of accuracy possible and to implement our model into practice in the real world.
All the models of Mobilenet, Dexpression, CNN, Densenet, and ResNet were evaluated.
The dcnn (ResNet) model was chosen because it had the highest training accuracy of all the models, and its validation accuracy was nearly 72 percent, which is comparable to CNN models.
As a result, we save this resnet model and use it to predict facial expressions.
Using streamlit, a front-end model was successfully created and ran on a local webserver.
The Streamlit web application has been successfully deployed on Amazon's AWS cloud platform.
