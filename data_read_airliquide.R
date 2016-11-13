#data vis from usb

library(data.table)
library(dplyr)
library(tidyr)
library(lubricate)
library(ggplot2)


#Currently, Airgas like many other small online businesses with little user activity on their search engines have no good way of evaluating the performance
#of their search algorithms,making it difficult for them to provide a customer experience that grows the more the customers use it or their customer base grows.
#more is better here and Airgas can surely handle more!

#The goal of this engine is to create a model for airgas that can be used to measure the relevance of search results.
#In doing so, we'll be helping enable naive customers to better sort out their seearch results for small business owners
#to match the experience provided by more resource rich competitors like Amazon.
#This engine will also provide airgas a model to test against. Given the queries and resulting product descriptions
#from their Google Ad campaign. This product will help airgas to evaluate the accuracy of their search algorithms.

# setwd("C:/Users/crc52_000/Desktop/HackathonDataSets/EasySearch/csv")
train <- read.csv("recItemsData.csv")  #readin GoogleKeywordVolumeandConversions-6

#some query analysis on the description clumn
library(ggvis)
library(tm)


######
#
# test code starts here  
#
######

library(readr)
library(Metrics)
library(tm)
library(SnowballC)
library(e1071)
library(Matrix)
library(SparseM)
library(caTools)

# Use readr to read in the training and test data
train <- read_csv("./train2.csv")
test  <-  read_csv("./test.csv")


#--------- remove html syntax in product description 
rm_garbage <- function(string){
  garbage <- c("<.*?>", "http", "www","img","border","style","px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed")
  for (i in 1:length(garbage)){
    string = gsub(garbage[i], "", string)
  }
  return (string)
}
train$product_description <-  lapply(train$product_description,rm_garbage)
test$product_description <-  lapply(test$product_description,rm_garbage)


#-----------restore dependent variables
test_id <- test$id
train_relevance <-  as.factor(train$median_relevance)
train_variance <-  train$relevance_variance
train$median_relevance = NULL
train$relevance_variance = NULL


#------------compute number words in independent variables 
n_words <- function(string){return(sapply(gregexpr("[[:alpha:]]+", string), function(x) sum(x > 0)))}

train_pt_words <-  apply(train[3],1,n_words)
test_pt_words <-  apply(test[3],1,n_words) 



#-- combine the sets' queries, p_d(s), p_t(s)
pt_words <- c(train_pt_words,test_pt_words)



#------------ bag of words 
DTM <-  function(corpus){
  dtm <- DocumentTermMatrix(corpus,control=list(tolower=TRUE,removePunctuation=TRUE,removeNumbers=TRUE,stopwords=TRUE,
                                               stemming=TRUE,weighting=function(x) weightTfIdf(x,normalize=T)))
  return (dtm)
}

combi <-  rbind(train,test)
all_text <-  Corpus(VectorSource(combi$query))
dtm <- DTM(all_text)
df_q <-  as.data.frame(as.matrix(dtm))
colnames(df_q) <-  paste("q_",colnames(df_q),sep="")




# Combine all columns into a single dataframe
combi <-  cbind(df_q,pt_words)  # combi = cbind(df_q,df_pt,df_pd,q_pt_match,pt_words,pt_pd_pred,pd_words,q_words) 

#-------------------------------- svm to predict relevance 
# Use naiveBayes to test various of ideas.. svm takes too long to run  
new_train <-  combi[1:10158,]
new_test <-  combi[10159:32671,]

#---separated the train set into train and test sets to play around with the "cost" variable in the svm. 
#---did fixed set cross validation to tune the model according to the in sample ScoreQuadraticWeightedKappa. 

# set.seed(1)
# spl= sample.split(train_relevance, SplitRatio = 0.7)



# sparse_train_train = Matrix(as.matrix(train_train),sparse=T)
# sparse_train_test = Matrix(as.matrix(train_test),sparse=T)
sparse_train <- Matrix(as.matrix(new_train),sparse=T)
sparse_test <- Matrix(as.matrix(new_test),sparse=T)

# model = svm(sparse_train_train,train_train_relevance, kernel="linear", cost=0.3)
# train_pred = predict(model, sparse_train_test)
# ScoreQuadraticWeightedKappa(train_pred, train_test_relevance,1,4)  

model <- svm(sparse_train, train_relevance, kernel="linear", cost=0.3)
pred <-  predict(model,sparse_test)
Newsubmission <-  data.frame(id=test_id, prediction = pred)
write.csv(Newsubmission,"model.csv",row.names=F)      #[1] 0.60718


#===============================================================
# Description: Script that generates a wordcloud out of trigrams extracted from title
#              corpous for query 'samsonite').
#              Demonstrates idea to use top trigram in title corpus as surrogate query.
# Input:       train.csv, test.csv
# Output:      wordcloud

library(wordcloud)
library(RWeka)
library(stringdist)
library(combinat)
library(readr)
library(qdap)
library(tm)

# Use readr to read in the training and test data
train <- read_csv("./train2.csv")
test  <- read_csv("./test.csv")
dset <- as.data.frame(rbind(train[,1:3], test[1:3]))

# do some cleaing on raw titles
dset$product_title <- tolower(dset$product_title) # convert to lower case
dset$product_title <- gsub("[ &<>)(_,.;:!?/-]+", " ", dset$product_title) # replace all punctuation and special characters by space
dset$product_title <- gsub("'s\\b", "", dset$product_title) # remove the 's
dset$product_title <- gsub("[']+", "", dset$product_title) # remove '
dset$product_title <- gsub("[\"]+", "", dset$product_title)   # remove "
dset$product_title <- gsub(paste("(\\b", paste(stopwords("english"), collapse="\\b|\\b"), "\\b)", sep=""), "", dset$product_title) # remove stopwords

# combine titles into one corpus (grouped by query)
queries <- unique(dset[,2])
q <- "samsonite"
titles <- dset$product_title[dset$query==q]
trigram <- NGramTokenizer(titles, Weka_control(min = 3, max = 3, delimiters = " \\r\\n.?!:"))
wd <- as.data.frame(table(trigram))
wd <- wd[order(-wd$Freq),]

# generate word cloud
pal2 <- brewer.pal(8,"Dark2")
# png("alt_query_wordcloud.png", width=12, height=8, units="in", res=300)
wordcloud(wd$trigram,wd$Freq, scale=c(1.8,0.2),min.freq=1,
          max.words=200, random.order=FALSE, rot.per=.15, colors=pal2, main="Blog")
#dev.off()

