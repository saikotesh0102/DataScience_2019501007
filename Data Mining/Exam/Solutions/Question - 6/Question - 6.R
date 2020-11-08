setwd("D:/Study/MSIT/2nd Year/DataScience_2019501007/Data Mining/Exam/Solutions/Question - 6")

data <- read.csv("Liver_data.csv", header = FALSE, col.names = c("mcv", "alkphos", "sgpt", "sgot", "gammagt", "drinks", "selector"))

x = data[ , 1 : 2]
plot(x, pch = 19,xlab = expression(x[1]), ylab = expression(x[2]))

fit<-kmeans(x, 4)
points(fit$centers,pch = 19,col = "blue",cex = 2)

library(class)
knnfit = knn(fit$centers,x,as.factor(c(-2, -1, 1, 2)))
points(x, col = 1 + 1 * as.numeric(knnfit),pch = 19)