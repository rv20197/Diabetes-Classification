#Read the file from the location where you have stored it.
diab_case=read.csv(file.choose(),header = T)

#It will show first 5 observations from the data frame
head(diab_case)

#Following are all the required libraries
library(dplyr)
library(caret)
library(ROCR)
library(VIM)
library(ggplot2)
library(class)
library(corrplot)
library(randomForest)
library(e1071)


#Creating subset to treat missing values
diab_case1=select(diab_case,Glucose,BloodPressure,SkinThickness,Insulin,BMI)
head(diab_case1)
diab_case1[diab_case1=='0']=NA#Replacing 0 with NA 
summary(diab_case1)#This will show the NA present in every individual variable
sum(is.na(diab_case1))#To check total number is NA present 
diab_case1=kNN(diab_case1,k=sqrt(nrow(diab_case1)))#KNN Imputation method to remove NA 
diab_case1=diab_case1[,1:5]#Removing Subset
#Replacing the treated variables with the untreated variables present in main data
diab_case$Glucose=diab_case1$Glucose
diab_case$BloodPressure=diab_case1$BloodPressure
diab_case$SkinThickness=diab_case1$SkinThickness
diab_case$Insulin=diab_case1$Insulin
diab_case$BMI=diab_case1$BMI


#Plotting count of variables for the datatypes present
datatype=lapply(diab_case,class)#Generate List
class(datatype)
datatype=data.frame(unlist(datatype))#It will convert list in to data frame
barplot(table(datatype),main = 'datatype',col='blue',ylab ='count of variables')



#Check Correlation and remove any one variable of the pair that has corelation value for than 0.9
pairs(diab_case,main="Scatter Plot",col='red')#Scatter Plot
diab_case_cor=cor(diab_case)#Creating correlatin matrix
heatmap(diab_case_cor,main='Heat Map')#Plotting heatmap
corrplot(diab_case_cor,method = 'number',type = 'upper',main='Co-Relation Plot')

diab_case$Outcome=factor(diab_case$Outcome)#It will convert variable Outcome into factor as it is required for classification purpose


#1)K-NEAREST NEAIGHBOUR(KNN) Classification
diab_case2=diab_case[,-9]#We'll be removing dependant variable Outcome as KNN works with numeric data by generating an another subset 
#Normalization/standardization/scaling of variables
normalize=function(x){
  return((x-min(x))/(max(x)-min(x)))
}
diab_case2=normalize(diab_case2)

#Holdout Cross Validation method 
#Divide data in training and testing 
indexdiab=sample(nrow(diab_case2),0.75*nrow(diab_case2))
train_diab=diab_case2[indexdiab,]#75% Training data for KNN
test_diab=diab_case2[-indexdiab,]#25% Testing Data for KNN

#Generate outcome vectors for traindata and testdata
ytrain=diab_case$Outcome[indexdiab]
ytest=diab_case$Outcome[-indexdiab]

#for generating KNN model
knnmodel=knn(train_diab,test_diab,k=sqrt(nrow(train_diab)),cl=ytrain);knnmodel
ytest=factor(ytest)
confusionMatrix(ytest,knnmodel)# Accuracy : 0.7448 of patients being correctly classified



#2) LOGISTIC REGRESSION
index=sample(nrow(diab_case),0.75*nrow(diab_case))#Here we are dividing data in 75:25 ratio
train_diab<-diab_case[index,]#Selecting 75% of the data
test_diab<-diab_case[-index,]#Selecting remaining 25% of the data
diabmodel<-glm(Outcome~.,data = train_diab,family = "binomial")#Generating Logistic Regression Model
summary(diabmodel)#It shows the summary of the generated model


#It will calculate the probability of patients being correct classified using Binary Logistic Regression
#Steps for ROCR Curve
train_diab$predprob<-fitted(diabmodel)#Predicting the probability values of patient being diabetic with the help of model we built  
head(train_diab)
pred<-prediction(train_diab$predprob,train_diab$Outcome)#Predicting the probabilistic values for the ROC curve
perf<-performance(pred,"tpr","fpr")#It will check the performance
plot(perf,colorize=T,print.cutoffs.at=seq(0.1,by=0.05),main='ROC Curve Plot for Training Data')#Plotting the curve
train_diab$predoutcome=ifelse(train_diab$predprob>0.35,1,0)#Filtering the values that are above the threshold value=0.35
train_diab$predoutcome=factor(train_diab$predoutcome)#Convert it into factor so that it can be compared with actual Outcome
confusionMatrix(train_diab$Outcome,train_diab$predoutcome)#Confusion matrix is generated for training data
#For train data   Accuracy : 0.7569 
diab_auctrain=performance(pred,"auc")#To find AUC(Area Under The Curve)
diab_auctrain@y.values#AUC Value is in y.values


#Process is same as that of training data except 
test_diab$predprob=predict(diabmodel,test_diab,type = "response")#type='response because predict function will generate sigmoid values and we are interested in values of probaility 
pred_test=prediction(test_diab$predprob,test_diab$Outcome)
perf_test<-performance(pred_test,"tpr","fpr")
plot(perf_test,colorize=T,print.cutoffs.at=seq(0.1,by=0.05))
test_diab$predoutcome=ifelse(test_diab$predprob<0.35,0,1)
test_diab$predoutcome=factor(test_diab$predoutcome)
confusionMatrix(test_diab$Outcome,test_diab$predoutcome)
diab_auctest=performance(pred_test,"auc")
diab_auctest@y.values#Area Under The ROC Curve for train data
#For test data  Accuracy : 0.7552 


#Now NULL all the following variables from the testing and training data as the are predictions and are not need for further classification techniques
train_diab$predoutcome=NULL
train_diab$predprob=NULL
test_diab$predprob=NULL
test_diab$predoutcome=NULL



#3)RANDOM FOREST
diab_caserf=diab_case#Generated seperate subset for random forest
set.seed(100)
Diab_RF=randomForest(Outcome~.,data=train_diab,ntree=10);Diab_RF#Generating random forest 
plot(Diab_RF)#This will plot the random forest graph
#It will calculate the probability of patients being correct classified using Random Forest
predrf_train=predict(Diab_RF,train_diab)
confusionMatrix(train_diab$Outcome,predrf_train)
#For train data  Accuracy : 0.9861
predrf_test=predict(Diab_RF,test_diab)
confusionMatrix(test_diab$Outcome,predrf_test)
#For test data  Accuracy : 0.75 



#4)NAIVE BAYES
naive_model_diab=naiveBayes(Outcome~.,data = train_diab);naive_model_diab#Method to build naive bayes model
#It will calculate the probability of patients being correct classified using Naive Bayes
predi_train_naive=predict(naive_model_diab,train_diab)
confusionMatrix(train_diab$Outcome,predi_train_naive)#Training Accuracy : 0.7743 
predi_test_naive=predict(naive_model_diab,test_diab)
confusionMatrix(test_diab$Outcome,predi_test_naive)# Testing Accuracy : 0.7604  


#5)SUPPORT VECTOR MACHINE(SVM)
svm_case_diab=svm(Outcome~.,data = train_diab,kernel='linear',scale = F)#Genrating SVM with scale=F as data is already normalized
#It will calculate the probability of patients being correct classified using SVM
predict_svm_casestudy=predict(svm_case_diab,train_diab)
confusionMatrix(train_diab$Outcome,predict_svm_casestudy)#  Accuracy : 0.7569
predict_svm_casestudy_train=predict(svm_case_diab,test_diab)
confusionMatrix(test_diab$Outcome,predict_svm_casestudy_train) #Accuracy : 0.7552  
