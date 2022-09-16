---
title: "Practical Machine Learning: Prediction Assignment"
author: "Jochen van Waasen"
date: '2022-09-015'
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(corrplot)
library(caret)
library(gbm)
```

## 1. Overview
In this project, we use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. The participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

The goal is to predict how well the six participants practiced their exercises.

## 2. Loading and preparing the data

```{r load data, warning=FALSE, message=FALSE, echo=TRUE}
dt_training <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"), strip.white = TRUE)
```

Quick data overview:
```{r}
dim(dt_training)
head(dt_training)
```

Data quality (identify columns with high percentage of NA data values):
```{r}
colMeans(is.na(dt_training))
```

If a column contains NA values it seems almost all the data in that column is not usable: > 97% NA per column. These column do not add value as the amount of data provided is too small.

Data cleaning: remove columns with > 97% NA
```{r}
dt_training = dt_training[,!sapply(dt_training, function(x) mean(is.na(x)))>0.97]
```

The quick data overview showed a lot of columns with empty entries also. Replace all empty values with NA and identify the percentage of empty (NA) values

```{r}
dt_training[dt_training == ''] <- NA
colMeans(is.na(dt_training))
```

The same as for the previous NA columns applies, > 97% of the values are empty (NA). To enhance dataset handling I remove columns with > 97% NA
```{r}
dt_training = dt_training[,!sapply(dt_training, function(x) mean(is.na(x)))>0.97]
```

After this cleaning the number of columns in the dataframe is reduced to 60.
```{r}
dim(dt_training)
head(dt_training)
```

The timestamp columns and the window columns do not add any insight as well.
```{r}
dt_training <- dt_training[,-(1:7)]
dim(dt_training)
```

Correlation Analysis

Let's get an idea of the how the variables influence each other.
```{r}
correlation_matrix <- cor(dt_training[, -53])
corrplot(correlation_matrix,  order = "FPC", type = "lower", method = "color", tl.cex = 0.6)
```

As correlation is weak we will not use principal components analysis.

## 3. Prediciton 

## Model Building

For the building the model I use 70% of the training dataset and the remaining 30% to test the model. 
```{r}
dt_training_partition  <- createDataPartition(dt_training$classe, p=0.7, list=FALSE)
dt_training_train <- dt_training[dt_training_partition, ]
dt_training_test  <- dt_training[-dt_training_partition, ]
```

Ensure reproducibility
```{r}
set.seed(8498)
```

I am going to use three models mentioned in the lecture from trees, random forest and boosting. For each model, I apply cross-validation (fitControl).

```{r}
fitControl <- trainControl(method='cv', number = 3)

model_rpart <- train(classe ~ ., data=dt_training_train, trControl=fitControl, method='rpart')
model_gbm <- train(classe ~ ., data=dt_training_train, trControl=fitControl, method='gbm', verbose = FALSE)
model_rf <- train(classe ~ ., data=dt_training_train, trControl=fitControl, method='rf')
```

Assessment

Out-of-sample error
```{r}
model_rpart$finalModel
model_gbm$finalModel
model_rf$finalModel
```

Random Forest has the highest accuracy.

Check the accuracy with the training data test set (30% not used for training)

```{r}
predRPART <- predict(model_rpart, newdata = dt_training_test)
```
Accuracy RPART
```{r}
confusionMatrixRPART <- confusionMatrix(predRPART, as.factor(dt_training_test$classe))
confusionMatrixRPART$overall[1]

predGBM <- predict(model_gbm, newdata = dt_training_test)
```
Accuracy GBM
```{r}
confusionMatrixGBM <- confusionMatrix(predGBM, as.factor(dt_training_test$classe))
confusionMatrixGBM$overall[1]

predRF <- predict(model_rf, newdata = dt_training_test)
```
Accuracy RF
```{r}
confusionMatrixRF <- confusionMatrix(predRF, as.factor(dt_training_test$classe))
confusionMatrixRF$overall[1]
```

## Verification

Let's use Random Forest which has shown the highest accuracy.

```{r}
dt_testing <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"), strip.white = TRUE)

predVerification <- predict(model_rf, dt_testing)
predVerification
```

## 4. Conclusion

Random Forest is the best fit for the data available.
