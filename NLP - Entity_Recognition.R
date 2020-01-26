Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre1.8.0_131')

library(lsa)
library(tm)
library(topicmodels)
library(openNLP)
library(openNLPdata)
library(NLP)
library(dplyr)
library(rJava)
library(RWeka)
library(RWekajars)
library(tm)
library(qdap)
library(corrplot)
library(philentropy) # Will be used Jaccard Distance

data<-read.csv(file.choose())
text<-as.character(data$text)

# Data Cleaning
corpus1<-Corpus(VectorSource(text)) 

corpus1 <- tm_map(corpus1, removeNumbers)
corpus1 <- tm_map(corpus1, removePunctuation)
corpus1 <- tm_map(corpus1 , stripWhitespace)
corpus1 <- tm_map(corpus1, content_transformer(tolower))
corpus1 <- tm_map(corpus1, removeWords, c(stopwords("english"),"the","am","gopdeb","gop"))
corpus1 <- tm_map(corpus1, stemDocument, language = "english")

unigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 3))
dtm<-DocumentTermMatrix(corpus1,control = list(tokenize=unigramTokenizer, 
                                               stopwords = c(stopwords("english"),"the","am","gopdeb","gop")))
dtm_keys<-inspect(dtm)

dtm_key1<-removeSparseTerms(dtm,sparse=0.9999)
inspect(dtm_key1)
tdm<-TermDocumentMatrix(corpus1,control = list(tokenize=unigramTokenizer, 
                                               stopwords = c(stopwords("english"),"the","am","gopdeb","gop")))
tdm_keys<-inspect(tdm)


terms1<-findFreqTerms(dtm,lowfreq=693)

associations<-findAssocs(dtm,"good",0.10)

freqr <- colSums(as.matrix(dtm))
ordr <- order(freqr,decreasing=TRUE)

freqr[head(ordr,20)]

freqr[tail(ordr,200)]


library(wordcloud)
wordcloud(names(freqr),freqr, min.freq=450,colors=brewer.pal(7,"Dark2"))


positives<-c("good","great","yoyo")
negatives<-c("nope","oops")

sentence<-c("This is great achievement","oops this is not good")

matches<-term_match1(sentence,c("great achievement","oops"))


term_match1<-function (text.var, terms, return.list = TRUE, apostrophe.remove = FALSE) 
{
  y <- rm_stopwords(text.var, stopwords = NULL, unlist = TRUE, 
                    strip = TRUE, unique = TRUE, apostrophe.remove = apostrophe.remove)
  x <- lapply(unlist(terms), function(z) {
    v <- term.find(y, mat = z, logic = TRUE)
    y[v]
  })
  names(x) <- unlist(terms)
  if (!return.list) {
    x <- sort(unique(unlist(x)))
  }
  x
}



score<-0

if(length(matches$good)>0){
  score<-1
}else if(length(matches$oops)>0)
{
  score<- -2
}


# Topic Modelling


dt_matrix1 <- removeSparseTerms(dtm, 0.99)
rowTotals <- apply(dt_matrix1 , 1, sum) 
dtm.new   <- dt_matrix1[rowTotals> 0, ]

lda <- LDA(dtm.new, k =5)

terms(lda)
topics(lda)

ctm<-CTM(dtm.new, k =5)
terms(ctm)
topics(ctm)
data<-as.matrix(dtm.new)

#store different distance matrices
eu <- dist(data, method="euclidean")
eudf<-data.frame(as.matrix(eu))
man <- dist(data, method="manhattan")
jac <- distance(data, method = "jaccard")
mandf<-data.frame(as.matrix(man))

write.csv(data.frame(as.matrix(jac)),"jac.csv")


tfidf<-weightTfIdf(dtm.new,normalize=TRUE)
tfidfframe<-data.frame(as.matrix(tfidf))

cosscores<-cosine(data)



library(vegan)
test<-data(varespec)
vare.dist <- vegdist(varespec, method = "jaccard")
p<-data.frame(as.matrix(vare.dist))


# Named Entity Recognition

personcount <- NULL
placecount <- NULL
organizationcount <- NULL

word_ann <- Maxent_Word_Token_Annotator()
sent_ann <- Maxent_Sent_Token_Annotator()

location_ann <- Maxent_Entity_Annotator(kind = "location",model="H:/Upworks/JayNandhita/NLP/es-ner-location.bin")
organization_ann <- Maxent_Entity_Annotator(kind = "organization",model="H:/Upworks/JayNandhita/NLP/en-ner-organization.bin")
person_ann <- Maxent_Entity_Annotator(kind = "person",model="H:/Upworks/JayNandhita/NLP/en-ner-person.bin")
pipeline <- list(sent_ann,word_ann,location_ann,
                 organization_ann, person_ann)
# No easy way to extract names entities from documents. 
# But the function below will do the trick.
entities <- function(doc, kind) {
  s <- doc$content
  a <- annotations(doc)[[1]]
  if(hasArg(kind)) {
    k <- sapply(a$features, `[[`, "kind")
    s[a[k == kind]]
  } else {
    s[a[a$type == "entity"]]
  }
}
for (i in 1: length(data$text)){
  tryCatch({
    
    # Pass list of annotator functions to the annotate() function and apply it to the bio variable. 
    bio_annotations <- annotate(data$text[i], pipeline)
    bio_doc <- AnnotatedPlainTextDocument(data$text[i], bio_annotations)
    # Extract all of the named entities using entities(bio_doc), and specific kinds of entities using the kind = argument. 
    # Get all the people.
    personname <- list(entities(bio_doc, kind = "person"))
    personname <- personname[[ !duplicated(personname)]]
    personcount[i] <- length(personname)
    # Get all the places.
    placename <- list(entities(bio_doc, kind = "location"))
    placename <- placename[[ !duplicated(placename)]]
    placecount[i]<- length(placename)
    # Get all the organizations.
    organizationname <- list(entities(bio_doc, kind = "organization"))
    organizationname <- organizationname[[ !duplicated(organizationname)]]
    organizationcount[i] <- length(organizationname)
    #print(organizationcount[i])
  }, warning = function(warning_condition) {
    
  }, error = function(error_condition) {
    
  }, finally={
    
  })
  
}

NER <- data.frame(cbind(personcount,placecount,organizationcount))



######POS  Tagging
p<-pos(data$text)
pos<-data.frame(p$POSfreq)

frames<-cbind(data,NER,pos)

hist(frames$personcount)
table(frames$organizationcount,frames$candidate)
