setwd("D:/Study/MSIT/2nd Year/DataScience_2019501007/Data Mining/DM Assignment2")

data <- read.csv("OH_house_prices.csv", header = FALSE)

head(data)

names(data)

median(data$V1)

mean(data$V1)