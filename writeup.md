---
title: "Exercise Form"
author: "Srujan Routhu"
date: "Sunday, September 21, 2014"
output: html_document
---

# Predicting the Form of Exercise

## Synopsis

In this article, we outline the process for predicting the form of the exercise being conducted by a subject based on the data extracted by using various hardware in exercise monitoring devices. After building and cross validating our model, we estimate the out of sample error to be 0.56 %.

More information pertaining to such a format of study can be found [here](http://groupware.les.inf.puc-rio.br/har)

## Data Importing and Processing

The training data used for this writeup was downloaded from [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv). After copying the data into our working folder we start the process of estimation.


```r
data <- read.csv("pml-training.csv", na.strings = c("NA", ""))

dim(data)
```

```
## [1] 19622   160
```

```r
str(data$classe)
```

```
##  Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```

```r
summary(data$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

We see that there are 5 different forms of exercise as denoted by the "classe" variable, with factor level A, B, C, D, E.

Next we split the data into two sets, one for training and the other for cross-validation to verify the performance of the model.


```r
set.seed(2906)
library(caret)

inTrain <- createDataPartition(y=data$classe, p=0.7, list=FALSE)

training <- data[inTrain, ]
valid <- data[-inTrain, ]
```

We have 159 variables available to predict the "classe" variable. We check these variables to see if there are any columns with a very high percentage of NA values, and remove them as their effect cannot be credibly quantified on the "classe" variable.


```r
check <- sapply(training, function(x) {sum(is.na(x))})
summary(check)
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##       0       0   13400    8410   13400   13400
```

```r
summary(as.factor(check))
```

```
##     0 13450 
##    60   100
```

From the above results it can be seen that 100 columns have an extremely high percentage of NA values. These columns can be ignored for this prediction model.


```r
bad <- names(check[check >= 1])
training <- training[, !names(training) %in% bad]

colnames(training)
```

```
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
##  [7] "num_window"           "roll_belt"            "pitch_belt"          
## [10] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
## [13] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [16] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [19] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [22] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [25] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [28] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [31] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [34] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [37] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [40] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [43] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [46] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [49] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [52] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [55] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [58] "magnet_forearm_y"     "magnet_forearm_z"     "classe"
```

The first seven variables are time and subject specific, and will not have any useful effect on the prediction model. Hence, these variables can also be removed.


```r
training <- training[, 8:60]
```

## Model Development

We will now build a model to predict the "classe" variable using the other variables. We use the random forest method for this prediction model.

The code chuck running the model prediction has been commented out after the model was saved to a file. This was done to avoid to singnificant amount of time the train function took to run.


```r
library(caret)
library(randomForest)

# fit <- train(classe~., method = "rf", data = training)
# saveRDS(fit, "fit.RDS")

model <- readRDS("fit.RDS")
```

## Validation

We now test the accuracy of the predictive model by validating it against the valid dataset we extracted out of the original data.


```r
validations <- predict(model, newdata=valid)
confusionMatrix(validations, valid$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    3    0    0    0
##          B    0 1135    1    0    0
##          C    0    1 1024    2    0
##          D    0    0    1  958    2
##          E    0    0    0    4 1080
## 
## Overall Statistics
##                                         
##                Accuracy : 0.998         
##                  95% CI : (0.996, 0.999)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.997         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.996    0.998    0.994    0.998
## Specificity             0.999    1.000    0.999    0.999    0.999
## Pos Pred Value          0.998    0.999    0.997    0.997    0.996
## Neg Pred Value          1.000    0.999    1.000    0.999    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.163    0.184
## Detection Prevalence    0.285    0.193    0.175    0.163    0.184
## Balanced Accuracy       1.000    0.998    0.999    0.997    0.999
```

```r
accuracy <- round(mean(validations == valid$classe)*100, 2)
error <- round(100 - accuracy, 2)

accuracy
```

```
## [1] 99.76
```

```r
error
```

```
## [1] 0.24
```

## Results

We see that the confusion matrix above shows a high value of accuracy. From the results displayed above, we estimate the out of sample error to have a value of 0.24 %.
