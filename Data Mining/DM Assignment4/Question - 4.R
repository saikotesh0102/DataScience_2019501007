setwd("D:/Study/MSIT/2nd Year/DataScience_2019501007/Data Mining/DM Assignment4")

data <- read.csv("test_data.csv", header = TRUE)

head(data)

library(caret)
library(rpart.plot)

x = data[ , 1 : 3]
y = data$X

model = rpart(y~., x , control = rpart.control(minsplit = 0,minbucket = 0,cp = -1, maxcompete = 0, maxsurrogate = 0, usesurrogate = 0, xval = 0, maxdepth = 5))

plot(model)
text(model)

rpart.plot(model)