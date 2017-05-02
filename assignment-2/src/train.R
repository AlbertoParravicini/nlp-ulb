library(readr)
library(rattle)					# Fancy tree plot
library(rpart.plot)	
# Classification Tree with rpart
library(rpart)

df <- read_csv("~/nlp-ulb/assignment-2/data/features.txt")

f <- lapply(FUN = as.factor, X = df)




# grow tree 
fit <- rpart(action ~ .,
             method="class", data=df, control=rpart.control(cp = 0, maxdepth = 10))

printcp(fit) # display the results 
plotcp(fit) # visualize cross-validation results 
summary(fit) # detailed summary of splits

# plot tree 
fancyRpartPlot(fit) 