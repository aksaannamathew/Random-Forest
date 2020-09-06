install.packages("randomForest")
library(readr)
library(randomForest)
library(caret)
install.packages("DataExplorer")
library(DataExplorer)

#Importing Data Set
Company <- read.csv("C:\\Users\\91755\\Desktop\\Assignment\\13 - Random Forest\\Company_Data.csv")
attach(Company)
View(Company)
head(Company)

#EDA
sum(is.na(Company))
summary(Company)
str(Company)

#Graphical Representation
plot(Company)
plot_correlation(Company)

#Converting Continous into Categorical Variable
Sales_cat <- ifelse(Company$Sales > 7.5, "High", "Low")
Company_sales <- data.frame(Sales_cat, Company)
Company_sales <- Company_sales[, -2]
head(Company_sales)
table(Company_sales$Sales_cat)

#Splitting Dataset
set.seed(123)
split <- createDataPartition(Company_sales$Sales_cat, p=0.7, list = F)
train_company <- Company_sales[split,] 
test_company <- Company_sales[-split,]
head(train_company)

#Model Building
model_1 <- randomForest(Sales_cat~., data = train_company)
model_1

#Graphical Representation
varImpPlot(model_1)
plot(model_1)
importance(model_1)
hist(treesize(model_1), main = "No.of Nodes of the Tree", col = "green")

#Evaluation
pred_1 <- predict(model_1, newdata = test_company)
pred_1
confusionMatrix(pred_1, test_company$Sales_cat)            #Accuracy = 78.99%

#Tune mtry
Tune <- tuneRF(train_company[,-1], train_company[,1], stepFactor = 0.5, plot = T, ntreeTry = 500, trace = T, improve = 0.05)

#Final Model
Model_final <- randomForest(Sales_cat~., data = train_company, mtry = 6, importance =T)
Model_final
varImpPlot(Model_final)
plot(Model_final)

#Evaluation
pred_final <- predict(Model_final, test_company)
confusionMatrix(pred_final, test_company$Sales_cat)         #Accuracy = 77.31%

#Model Using Train Function
set.seed(123)
model_2 <- train(Sales_cat~., data=train_company, method ="rf", trControl = trainControl(method = "repeatedcv", n=10))
model_2

pred_2 <- predict(model_2, test_company, type = "raw")
confusionMatrix(pred_2, test_company$Sales_cat)           #Accuracy = 75.63

#Bagging Algorithm
trainControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
seed <- 7
fit_tree <- train(Sales_cat~., data = train_company, method="treebag", metric = "Accuracy", 
                  trainControl=trainControl)

#Evaluation
pred_bag <- predict(fit_tree, test_company, type = "raw")
pred_bag
confusionMatrix(pred_bag, test_company$Sales_cat)    #Accuracy = 76.47
