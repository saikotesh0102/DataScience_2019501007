setwd("D:/Study/MSIT/2nd Year/DataScience_2019501007/Data Mining/DM Assignment2")

data <- read.csv("football.csv", header = TRUE)

names(data)

head(data)

str(data)

plot(X2004.Wins ~ X2003.Wins, data = data, main = "Scatter Plot : 2003 Wins vs 2004 Wins")

cor(data$X2003.Wins, data$X2004.Wins)

data1 <- data

data1 <- data1[,3] * 2

head(data1)

cor(data$X2003.Wins, data1)