library(readr)
library(randomForest)
library(caret)
library(DataExplorer)

#Importing Data
Fraud_Data <- read.csv("C:\\Users\\91755\\Desktop\\Assignment\\13 - Random Forest\\Fraud_check.csv")
attach(Fraud_Data)
head(Fraud_Data)

#EDA
sum(is.na(Fraud_Data))
summary(Fraud_Data)
str(Fraud_Data)

#Graphical Representation
plot(Fraud_Data)
plot_correlation(Fraud_Data)

#Converting Continous Variable into Categorical
Tax_income_cat <- ifelse(Fraud_Data$Taxable.Income<=30000, "Risky", "Good")
Fraud_cat <- data.frame(Tax_income_cat, Fraud_Data)
Fraud_cat <- Fraud_cat[, -4]
head(Fraud_cat)
table(Fraud_cat$Tax_income_cat)

#Data Splitting
set.seed(123)
split <- createDataPartition(Fraud_cat$Tax_income_cat, p=0.7, list = F)
Train_Fraud <- Fraud_cat[split,]
Test_Fraud <- Fraud_cat[-split,]
head(Train_Fraud)

#Model Buildind
Model_1 <- randomForest(Tax_income_cat~., data = Train_Fraud)
Model_1

#Graphical Representation
plot(Model_1)
importance(Model_1)
varImpPlot(Model_1)
hist(treesize(Model_1), main = "No. of Nodes of the Tree", col = "blue")

#Evaluation
pred_1 <- predict(Model_1, Test_Fraud)
pred_1
confusionMatrix(pred_1, Test_Fraud$Tax_income_cat)           #Accuracy = 77.65

##Tune mtry
Tune <- tuneRF(Train_Fraud[, -1], Train_Fraud[, 1], stepFactor = 0.5, plot = T, ntreeTry = 50, trace = T, improve = 0.05)

#Final Model Building
Model_final <- randomForest(Tax_income_cat~., data = Train_Fraud, mtry=1, importance=T)
Model_final
varImpPlot(Model_final)

pred_final <- predict(Model_final, Test_Fraud)
confusionMatrix(pred_final, Test_Fraud$Tax_income_cat)      #Accuracy = 79.33

#Model Building Using Train Function
set.seed(222)
model_rf <- train(Tax_income_cat~., data = Train_Fraud, method="rf", trControl=trainControl(method = "cv", n=10))
model_rf

plot(model_rf)

pred_rf <- predict(model_rf, Test_Fraud, type = "raw")
confusionMatrix(pred_rf, Test_Fraud$Tax_income_cat)        #Accuracy = 79.33

#Bagging Algorithm
set.seed(222)
model_treebag <- train(Tax_income_cat~., data = Train_Fraud, method="treebag", metric="Accuracy", 
                       trControl=trainControl(method = "repeatedcv", n=10, repeats = 3))
pred_treebag <- predict(model_treebag, Test_Fraud, type = "raw")
pred_treebag
confusionMatrix(pred_treebag, Test_Fraud$Tax_income_cat)  #Accuracy = 71.51
