setwd("D:/Study/MSIT/2nd Year/DataScience_2019501007/Data Mining/Exam/Solutions/Question - 3")

# install.packages("caret")
library(caret)
library(rpart.plot)

data <- read.csv("lenses.data.csv", header = FALSE, col.names = c("1", "2", "3", "4", "5", "Label"))

str(data)

head(data)

x = data[ , 2 : 5]
y = as.factor(data$Label)
# As factor is used to convert a numerical data to categorical data

model = rpart(y~., x , control = rpart.control(minsplit = 0,minbucket = 0,cp = -1, maxcompete = 0, maxsurrogate = 0, usesurrogate = 0, xval = 0, maxdepth = 5))

plot(model)
text(model)

rpart.plot(model)

#Information Gain
sum(y == predict(model, x, type = "class"))/ length(y)

#misclassification error
1 - sum(y == predict(model, x, type = "class"))/ length(y)

model1 = rpart(y~., x, control = rpart.control(minsplit = 0,minbucket = 0,cp = -1, maxcompete = 0, maxsurrogate = 0, usesurrogate = 0, xval = 0, maxdepth = 7))

plot(model1)
text(model1)

rpart.plot(model1)

#Information Gain
sum(y == predict(model1, x, type = "class"))/length(y)

#miscalassification error
1 - sum(y == predict(model1, x, type = "class"))/length(y)