setwd("D:/Study/MSIT/2nd Year/DataScience_2019501007/Data Mining/DM Assignment2")

data <- read.csv("myfirstdata.csv", header = FALSE)

head(data) # Gives us the head of the data

names(data) #Column Names

str(data) #description of each column of dataset

dim(data) #dimensions of data Rows by Columns

plot(data[ , 1]) #plots the data of only first column

plot(data[ , 2]) #plots the data of only second column

data1 <- read.csv("myfirstdata.csv", header = FALSE)

head(data1)