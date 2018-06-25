'''
To compare the classifiers I used the following idea
http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
'''
import pandas
from wordcloud import WordCloud
import numpy
import seaborn
import matplotlib
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn import metrics
import nltk
#nltk.download('all')
from nltk.corpus import stopwords

'''
Function for reading the CSV file
I used the pandas library that reads the CSV file and leaves it in a tables format,
making it easier to choose between spam and normal messages
'''
def reading_csv():
	file_sms = pandas.read_csv('./database_total/spam.csv', encoding='latin-1')
	file_sms.head()
	file_sms = file_sms.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
	file_sms = file_sms.rename(columns={'v1':'label','v2':'message'})
	return file_sms

'''
For the preaching I used the idea presented in the header of this code, where I made the comparison of the accuracy
between the classifiers for the messages sent. I used the scikit-learn library for preaching.
'''
def predicao():
	#Text Processing
	file_sms = reading_csv()
	training_message,test_message,label_training,label_test = train_test_split(file_sms["message"],file_sms["label"], test_size = 0.2, random_state = 10)
	vect = CountVectorizer()
	vect.fit(training_message)
	training_message_new = vect.transform(training_message)
	test_message_new = vect.transform(mensagem_teste)

	#Performing Prediction
	svc = SVC(kernel='sigmoid', gamma='auto')
	knc = KNeighborsClassifier(n_neighbors=50)
	mnb = MultinomialNB(alpha=1.0)
	dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111)
	lrc = LogisticRegression(solver='liblinear', penalty='l1')
	rfc = RandomForestClassifier(n_estimators=30, random_state=111)
	abc = AdaBoostClassifier(n_estimators=60, random_state=111)
	bc = BaggingClassifier(n_estimators=10, random_state=111)
	etc = ExtraTreesClassifier(n_estimators=10, random_state=111)
	clfs = {'SVC' : svc,'KN' : knc, 'NB': mnb, 'DT': dtc, 'LR': lrc, 'RF': rfc, 'AdaBoost': abc, 'BgC': bc, 'ETC': etc}
	list_results = []
	for k,v in clfs.items():
		v.fit(training_message_new, label_training)
		pred = v.predict(test_message_new)
		list_results.append((k, [accuracy_score(label_test,pred)]))
	result = pandas.DataFrame.from_items(list_results,orient='index', columns=['Score'])
	print(result)
	result.plot(kind='bar', ylim=(0.8,1.0), figsize=(13,9),color='blue', align='center')
	plot.xticks(numpy.arange(9), result.index)
	plot.ylabel('Accuracy')
	plot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plot.savefig('resultado_classificadores.png')

	#matrix confusion NaiveBayes
	predict_naive = mnb.predict(test_message_new)
	print("\n")
	print("Matriz de Confusao Naive Bayes")
	print(metrics.confusion_matrix(label_test,predict_naive))


'''
For the processing and visualization of the text I used the library NLTK that is proper for word processing, 
I also used the idea of pandas tables. 
I also used the seaborn tool to perform a media to get the message sizes and generate a graph
'''
def visualizacao():
	file_sms = reading_csv()
	groupby_label = file_sms.groupby('rotulo').describe()
	print(groupby_label)
	file_sms["length"]=file_sms["message"].apply(len)
	file_sms['label_number'] = file_sms.rotulo.map({'ham':0, 'spam':1})
	seaborn.kdeplot(file_sms[file_sms["label_number"]==0]["length"],shade=True,label="ham")
	plot.xlabel("length")
	seaborn.kdeplot(file_sms[file_sms["label_number"]==1]["length"],shade=True,label="spam")
	plot.savefig('length_spam_ham.png')
	textos_ham =  ''
	textos_spam = ''
	for value in file_sms[file_sms.rotulo_numero==0].message:
		texto = value.lower()
		tokens = nltk.word_tokenize(texto)
		for words in tokens:
			textos_ham = textos_ham + words + ' '
    
	for value in file_sms[file_sms.rotulo_numero==1].message:
		texto = value.lower()
		tokens = nltk.word_tokenize(texto)
		for words in tokens:
			textos_spam = textos_spam + words + ' '

	spam_wordcloud = WordCloud(width=300, height=200).generate(textos_spam)
	ham_wordcloud = WordCloud(width=300, height=200).generate(textos_ham)
	plot.figure( figsize=(10,8), facecolor='k')
	plot.imshow(spam_wordcloud)
	plot.axis("off")
	plot.tight_layout(pad=0)
	plot.savefig('spamText.png')
	plot.figure( figsize=(10,8), facecolor='k')
	plot.imshow(ham_wordcloud)
	plot.axis("off")
	plot.tight_layout(pad=0)
	plot.savefig('hamText.png')
	

if __name__ == '__main__':
	reading_csv()
	visualization()
	predicao()