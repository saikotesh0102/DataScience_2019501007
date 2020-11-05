setwd("D:/Study/MSIT/2nd Year/DataScience_2019501007/Data Mining/DM Assignment2")

data <- read.csv("CA_house_prices.csv", header = FALSE)

data1 <- read.csv("OH_house_prices.csv", header = FALSE)

dim(data)

dim(data1)

boxplot(data, col = "blue", main = "Sai Koteswara Rao's Box Plots")

boxplot(data1, col = "green", main = "Sai Koteswara Rao's Box Plots")

hist(data$V1, main = "Sai Koteswara Rao's Hist Plots")

hist(data1$V1, main = "Sai Koteswara Rao's Hist Plots")

hist(data$V1,breaks=seq(from = 0,to = 3500000,by = 500000),col = c("green","red","blue","yellow","orange"),main = "Frequency Histogram of California Houses", xlab = "California Houses Prices in thousands", ylab = "frequency")

plot(ecdf(data[, 1]), verticals = TRUE, do.p = FALSE, main = 'ECDF for House prices', xlab = 'Prices(in Thousands)', ylab = "Frequency")

lines(ecdf(data1[, 1]), verticals = TRUE, do.p = FALSE, col.h = 'red', col.v = 'red', lwd = 4)

legend(2100, 0.6, c("CA Houses", "OHIO Houses"), col = c('black', 'red'), lwd = c(1, 4))
