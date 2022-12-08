# NLP Inference Classification

## Abstract
A model has the ability to model narratives if it could infer implicit participant states. This project aims to build and train a system that better models the narratives of a story in order to be able to draw inferences or classify hypotheses. This project is aimed at addressing model inferability as proposed by the authors of the paper titled "PASTA: A Dataset for Modeling Participant States in Narratives".

## Introduction
This project aims to build, train and evaluate NLP-based transformer models on the ability to classify if the given inference statement is warranted by the given story. This requires the models to have an understanding of the world state for the given story in order to be able to predict whether the world state is consistent with the inference statement.

The authors of the original paper that proposed the inference task have evaluated transformer models such as BERT-base, RoBERTa-base, T5-base, and T5-large on the Participant States (PASTA) dataset. The results were sub-par when compared to human-level accuracy. But this sets up a good baseline to improve upon, which is the primary objective of this project.

The main evaluation criteria for this task is to measure the model performance with evaluation metrics such as Accuracy, Precision, Recall, F1-score, and AUC-ROC score. We would also compare it against the baseline Accuracy scores.


## Project outcomes

- We implemented various transformer models such as BERT-base, RoBERTa-base, T5-base, and T5-large for inference classification using the PASTA dataset.

- We improved the performance of the models by employing methods such as using model-specific embeddings and tokenizers, data pre-processing, data augmentation, and hyperparameter tuning, as seen through the significant improvement in the AUC-ROC metric.

- We also experimented with training the models on TPUs to significantly boost the train speed, thereby improving the rate at which we are able to experiment with different hyperparameters.

- Between the models, we observe that the T5-large performs the best among all the four models, and RoBERTa performs the worst.


## Task

The task of this project is to train a model that effectively classifies whether a given assertion is warranted by the given story. 


## Dataset

We used the Participant States (PASTA) dataset for this task. PASTA dataset contains a list of story lines, their corresponding assertion statement, a mod assertion statement, and the same story lines modified to warrant the mod assertion statement.

We constructed the dataset by merging the story lines of both, original and mod stories, and taking all the combinations of stories and lines for constructing 2 instances of positive and negative examples for each data row in the original dataset.
Approach

We set up and trained multiple transformer model architectures including BERT, RoBERTa, T5-base, and T5-large to replicate the baseline results, as mentioned in the paper.

Specifically, we focused on two aspects of model training that could significantly improve the performance of the models.

- Using model-specific embeddings and tokenizers: allows the utilization of the existing pre-trained model weights, which could then be fine-tuned on the PASTA dataset.

- Data pre-processing: We employed a variety of text-specific pre-processing methods such as removing special characters, and data augmentation where we replace some of the words with synonyms to generate more training data, etc.

- Hyperparameter tuning: We experimented with different hyperparameters such as the max length limit of tokens, number of epochs, learning rate, optimizer, batch size, etc. We also employed different callback methods such as ModelCheckpoint, EarlyStopping, and ReduceLR.

On a different aspect, we also aimed to improve the model training speed and efficiency. In order to achieve this, we employed the use of TPU cores provided by Kaggle (a platform for machine learning competitions). Specifically, we used 8 TPU cores to train multiple batches of training data parallelly, with a significant boost to the training speed per core.


## Evaluation measures

We used a variety of evaluation metrics to understand the performance of the models. The following are the relevant ones:

- Accuracy: Number of correct predictions by the size of the entire test set.

- Precision: Number of correct positive predictions by the total number of positive predictions.

- Recall: Number of positive predictions by the total number of positives in the test set.

- F1 score: Harmonic mean of precision and recall.

- AUC-ROC score: Area under the curve of Receiver operating characteristic graph that plots True Positive Rate (TPR) vs False Positive Rate (FPR).


## Results

We were able to improve the model’s performance by a significant amount using the above-mentioned techniques. This can be inferred from the high AUC-ROC score of the models, specifically that of the T5-large model.


## Links

Dataset link: https://www.kaggle.com/datasets/akashsuper2000/participant-states/

Model weights link: https://www.kaggle.com/datasets/akashsuper2000/pasta-dataset/

PASTA paper link: https://arxiv.org/abs/2208.00329
