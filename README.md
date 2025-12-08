# File descriptions

## Data 

***comments.json:*** large text file acquired from Index of /old/SARC/2.0/main. This is the file used to extract response and any other necessary text, which will be used to get our embeddings.  

***labeled_paired_test_embeddings:*** text embeddings retrieved from BERT with comment context; this is what we we will test on our model  

***labeled_paired_train_embeddings:*** text embeddings retrieved from BERT with comment context; this is what we we will train on our model  

***responses_and_comments_flat_test.csv:*** extracted from comments; contains comments and responses to be used for testing  

***responses_and_comments_flat_train.csv:*** extracted from comments; contains comments and responses to be used for training  

***responses_flat_test.csv:*** responses retrieved from from test_small_comments_fixed.json using response ids in test-balanced.csv, formatted as: response_id,response_text,label  

***responses_flat_train.csv:*** responses retrieved from from test_small_comments_fixed.json using response ids in train-balanced.csv, formatted as: response_id,response_text,label  

***strip_responses_test.csv:*** responses retrieved from comments disregarding punctuation; for testing  

***strip_responses_train.csv:*** responses retrieved from comments disregarding punctuation for training  

***stripped_test_embeddings_mean.npz:*** embedding retrieved from BERT on stripped responses; for testing  

***stripped_train_embeddings_mean.npz:*** embedding retrieved from BERT on stripped responses; for training 

***test_embeddings_mean.npz:*** BERT embeddings on regular responses using mean pooling; for testing  

***test_small_comments_fixed.json:*** shrunk big comments.csv to contain only response information, which is later formatted in responses_flat_test.csv; used test-balanced.csv to get appropriate response ids; for testing  

***test-balanced.csv:*** sequence file with testing data (20%); contains necessary post, comment, responses ids, as well as sarcasm scores (0/1) for the two given responses; sourced from Index of /old/SARC/2.0/main  

***train_embeddings_mean.npz:*** BERT embeddings on regular responses using mean pooling; for testing  

***train_small_comments_fixed.json:*** shrunk big comments.csv to contain only response information, which is later formatted in responses_flat_train.csv; used train-balanced.csv to get appropriate response ids; for training  

***train-balanced.csv:*** sequence file with training data (80%); contains necessary post, comment, responses ids, as well as sarcasm scores (0/1) for the two given responses; sources from Index of /old/SARC/2.0/main  


# Scripts

## Data process

***get_embeddings.py:*** generate BERT embeddings to be later used to pass through our model; generates Nx768 dimensional embeddings  

***real_flattener.py:*** extracts responses from smaller comments file into necessary format for embedding generation  

***shrinkcomments.py:*** shrinks large comments.csv file into viewed csv file for responses and comments based on ids  

***stripped_punctuation.py:*** strips response text of any punctuation (except commas and spaces); made to later make comparison in punctuation and semantic relation  

## Other scripts  

***model_trainer.py:*** our neural network structure that trains our model on the data  

***one_hot_baseline.py:*** simple baseline model to compare our model to; one-hot encoder  

***initial_model_trainer.py:*** our initial neural network structure that trains our model on the data  

# Figures

***[model]_loss_graph*** Training and validation loss across epochs for each model. Each was created in its respective trainer file using a variation of the following code block.

```python
#loss plot
plt.plot(history.history["loss"], label="Train Loss", color='darkorange')
plt.plot(history.history["val_loss"], label="Val Loss", color = 'darkgreen')
plt.title("Final Model Loss")
plt.xlabel("Number of Epochs")
sns.despine()  #remove spines from graph
plt.ylabel("Loss")
plt.legend()
plt.show()
```

***final_calibration_curve*** Calibration curve for our final model, showing the relationship between predicted probabilities and actual sarcasm frequencies. The curve was generated in the model_trainer.py file using the following code block.
```python
# calibration curve
# numerical prediction and targets _> check model accuracy scores against different decision thresholds
# -> for diff sensitivities of the output 
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# predicted probabilities on test set
y_prob = model.predict(x_test, verbose=0).ravel()  # shape (N_test,)

# compute calibration curve
prob_true, prob_pred = calibration_curve(
    y_test, y_prob,
    n_bins=10,        # number of bins
    strategy='quantile'  # each bin has ~same # of samples
)

# plot reliability (calibration curve) diagram
plt.figure()
plt.plot(prob_pred, prob_true, marker="o", linewidth=1, label="Model")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
plt.xlabel("Predicted probability")
plt.ylabel("Observed fraction of positives")
plt.title("Calibration curve (reliability diagram)")
plt.legend()
plt.grid(True)
plt.show()
```

***[model]_confusion_matrix*** Confusion matrix for each model for sarcasm classification on the test set using a probability threshold of 0.5 for the baseline and initial models, and 0.4 for our final model in accordance with the calibration curve. Each was created in its respective trainer file using a variation of the following code block.

```python
#confusion matrix
y_pred = (model.predict(x_test) > 0.4).astype(int)  #threshold of 0.4 based off of calibration curve
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',  
            xticklabels=['Non-Sarcastic', 'Sarcastic'],
            yticklabels=['Non-Sarcastic', 'Sarcastic'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Final Model Confusion Matrix")
plt.show()
```

***[model]_prob_distribution*** Histogram of predicted sarcasm probabilities for each model's test set. Each was created in its respective trainer file using a variation of the following code block.
```python
#probability histogram
y_prob = model.predict(x_test).ravel() #.ravel() flattens multidimensional data
plt.hist(y_prob, bins=10, color='darkgreen')
plt.title("Final Model Predicted Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
sns.despine()  #remove spines from graph
plt.show()
```

***data_flow_chart*** Flow chart created in BioRender to show the path our data takes through the project.

