from implement_word2vec import word2vec
from preprocessing_data import preprocessing, prepare_data_for_training
from nltk.corpus import stopwords

corpus = ""
corpus += "The earth revolves around the sun. The moon  revolves around the earth"
epochs = 1000

training_data = preprocessing(corpus)
w2v = word2vec()

prepare_data = prepare_data_for_training(training_data, w2v)
w2v.train(epochs)

print(w2v.predict("around", 2))