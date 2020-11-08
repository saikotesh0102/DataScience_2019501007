setwd("D:/Study/MSIT/2nd Year/DataScience_2019501007/Data Mining/Exam") 
#sets the working directory to our required path

data <- read.csv("BSE_Sensex_Index.csv", header = TRUE)
#Read the CSV and assign the data to variable data 

#dim(data)

#head(data)
#prints out the head of the data frame

data$Date <- as.Date(data$Date,format= "%m/%d/%Y")
diff_Date <- c()
for(i in 1:(length(data$Date) - 1)){
  diff <- difftime(data[i + 1, 1], data[i, 1], units = "days")
  diff_Date <- append(diff_Date,as.numeric(diff, units = "days")) 
}
diff_Date <- append(diff_Date, 0)

diff_Open <- diff(data$Open)
diff_Open <- append(diff_Open, mean(data[c(-5 : -2), 2]))

#length(diff_Open)

diff_High <- diff(data$High)
diff_High <- append(diff_High, mean(data[c(-5 : -2), 3]))

diff_Low <- diff(data$Low)
diff_Low <- append(diff_Low, mean(data[c(-5 : -2), 4]))

diff_Close <- diff(data$Close)
diff_Close <- append(diff_Close, mean(data[c(-5 : -2), 5]))

diff_Volume <- diff(data$Volume)
diff_Volume <- append(diff_Volume, mean(data[c(-5 : -2), 6]))

diff_Adj.Close <- diff(data$Adj.Close)
diff_Adj.Close <- append(diff_Adj.Close, mean(data[c(-5 : -2), 7]))

#class(diff_Open)

#length(diff_Open)

#length(diff_Close)

#length(diff_High)

#length(diff_Volume)

#length(diff_Adj.Close)

#length(diff_Low)

#print(dim(data.frame(diff_Open)))

#print(dim(data))

data <- cbind(data, data.frame(diff_Date, diff_Open, diff_High, diff_Low, diff_Close, diff_Volume, diff_Adj.Close))

#dim(data)

library(dplyr)
#installs the dpylr package

df1 <- sample_n(data, 1000, replace = TRUE)
#Created a sample of 1000 observations with replacement

#dim(df1)

#head(df1)

df2 <- sample_n(data, 3000, replace = TRUE)
#Created a sample of 3000 observations with replacement

#dim(df2)

#head(df2)

summary(df1)

summary(df2)

summary(data)

boxplot(data$Open, data$High, data$Close, data$Low,
        main = "Multiple boxplots for comparision",
        col = c("orange","red", "blue", "green"),
        names = c("Open", "High", "Close", "Low"))

hist(data$Close, 
     main = "Frequency plot for Close",
     xlim = c(0, 16000),
     xlab = "Close Values", 
     ylab = "Frequency",
     col = c("darkmagenta"))
