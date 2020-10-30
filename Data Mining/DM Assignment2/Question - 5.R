setwd("D:/Study/MSIT/2nd Year/DataScience_2019501007/Data Mining/DM Assignment2")

data <- read.csv("twomillion.csv", header = FALSE)

head(data)

#sample(data, 1, replace = TRUE)

str(data)

dim(data)

#library(dplyr)

myData_sample_10000 <- sample_n(data, 10000) # Sampling 10,000 lines

dim(myData_sample_10000)

max(myData_sample_10000)

min(myData_sample_10000)

var(myData_sample_10000)

mean(myData_sample_10000)

quantile(myData_sample_10000$V1,  probs = c(0.25))

summary(myData_sample_10000)

summary(data)

write.csv(myData_sample_10000, file = "D:/Study/MSIT/2nd Year/DataScience_2019501007/Data Mining/DM Assignment2/Sample_data.csv")