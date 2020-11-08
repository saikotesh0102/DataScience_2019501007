setwd("D:/Study/MSIT/2nd Year/DataScience_2019501007/Data Mining/Exam/Solutions/Question - 2")

library(arules)
library(arulesViz)
library(RColorBrewer)

# data <- data.frame (item_1  = c("A", "A", "D", "B", "A", "A", "B", "A", "A", "D"),
#                   item_2 = c("B", "C", "E", "C", "B", "B", "D", "B", "D", "E"),
#                   item_3 = c("C", "D", "", "E", "D", "", "E", "D", "", ""),
#                   item_4 = c("D","", "", "", "E", "", "", "", "", ""),
#                   item_5 = c("E", "", "", "", "", "", "", "", "", ""))

#The read.transactions() function read the file csv file and convert it to a transaction format
#Parameters: Transaction file: name of the csv file
#rm.duplicates : to make sure that we have no duplicate transaction entried
#format : basket (row 1: transaction ids, row 2: list of items)
#sep: separator between items, in this case commas
#cols : column number of transaction IDs
data <- read.transactions(file ="Apriori.csv", rm.duplicates = TRUE, format = "basket", sep = ",", cols = 1)

#getting rid of unnecessary quotes in transactions
data@itemInfo$labels <- gsub("\"","",data@itemInfo$labels)

# data <- read.csv("Apriori.csv", header = TRUE)

# head(data, 10)

rules <- apriori(data, parameter = list(supp = 0.3))
inspect(rules)

itemFrequencyPlot(data, topN = 5,
                  col = brewer.pal(8,'Pastel2'), 
                  main = "Absolute Item Frequency Plot")

plot(rules)
plot(rules, method = "graph",  engine = "htmlwidget")
