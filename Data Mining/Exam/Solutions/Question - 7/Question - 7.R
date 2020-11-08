setwd("D:/Study/MSIT/2nd Year/DataScience_2019501007/Data Mining/Exam/Solutions/Question - 7")

data = read.csv("Liver_data.csv", header = FALSE, col.names = c("mcv", "alkphos", "sgpt", "sgot", "gammagt", "drinks","selector"))

x = data[ , 1 : 5]
y = data[ , 6]
fit = kmeans(x , 4)
library(class)
knnfit = knn(fit$centers, x, as.factor(c(-2,-1,1,2)))
error = 1-sum(knnfit == y)/length(y)
print(error)