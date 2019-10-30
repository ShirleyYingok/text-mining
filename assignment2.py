import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import string

# Focus on "Joker" (2019)
url = 'https://www.imdb.com/title/tt7286456/reviews?ref_=tt_urv'
res = requests.get(url)
soup = BeautifulSoup(res.text, features='html.parser')

# Extract the link leading to the page containing everything available 
main_content = urljoin(url,soup.select(".load-more-data")[0]['data-ajaxurl'])  
# print(main_content)
response = requests.get(main_content)
broth = BeautifulSoup(response.text, features='html.parser')

# Write first 25 reviews data into a file
with open('review_output.txt','w') as f_out:
    for item in broth.select(".review-container"):
        review = item.select(".text")[0].text
        f_out.write(review)


def process_file (filename):
    '''
    Takes an argument: filename: string
    Process each word: remove punctuation and convert to lowercase
    Returns: a dictionary that map from each word to the number of times it appears.
    '''
    d = {}
    f = open(filename)
    strippables = string.punctuation + string.whitespace 

    for word in f.read().split():
        # print(word)
        word_lower = word.lower()
        # print(word_lower)
        clean_word = word_lower.strip(strippables) 
        # print(clean_word)
    
        d[clean_word] = d.get(clean_word,0) + 1
    
    return d


# Word Frequency 
def most_common(d):
    '''
    Takes an argument: d: a dictionary that map from word to frequency
    Makes a list of word-freq pairs in descending order of frequency.
    Returns: list of (frequency, word) pairs
    '''
    l = []
    for key, value in d.items():
        l.append((value, key))
    l.sort(reverse = True)
    return l


def print_most_common(d, num=10):
    '''
    Takes two arguments:
    the first is d: dictionary that map from word to frequency
    the second is num: number of pairs to print
    Prints the most commons words in pair (frequency,word)
    '''
    hist_common = most_common(d)
    for pair in hist_common[:num]:
        print (pair)


def main():
    # print(process_file('review_output.txt'))
    word_dictionary = process_file('review_output.txt')
    # print(most_common(word_dictionary))
    print_most_common(word_dictionary)


if __name__ == '__main__':
    main()




# Natural Lanuage Processing

# Sentiment Analysis
import nltk
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from imdbpie import Imdb

imdb = Imdb()

# print all movies that contain the word "Joker"
# print(imdb.search_for_title("Joker"))

# Focus on "Joker" (2019), obtain "imdb_id"
# print(imdb.search_for_title("Joker")[0]['imdb_id'])

reviews = imdb.get_title_user_reviews("tt7286456")

# print(reviews)
# print(reviews['reviews'])
# print(reviews['reviews'][0]['reviewText'])

review1 = reviews['reviews'][0]['reviewText']
review2 = reviews['reviews'][1]['reviewText']
review3 = reviews['reviews'][2]['reviewText']
score1 = SentimentIntensityAnalyzer().polarity_scores(review1)
score2 = SentimentIntensityAnalyzer().polarity_scores(review2)
score3 = SentimentIntensityAnalyzer().polarity_scores(review3)
# print(score1)
      
def sentiment_analysis(d):
    '''
    Takes in an argument: d: a dictionary that contains pos, neg, neu, and compound scores 
    Prints the sentiment analysis result
    '''
    print("Overall sentiment dictionary is : ", d) 
    print("sentence was rated as ", d['neg']*100, "% Negative") 
    print("sentence was rated as ", d['neu']*100, "% Neutral") 
    print("sentence was rated as ", d['pos']*100, "% Positive") 

    print("Sentence Overall Rated As", end = " ") 
    
    # decide sentiment as positive, negative and neutral 
    if d['compound'] >= 0.05 : 
        print("Positive") 
    
    elif d['compound'] <= - 0.05 : 
        print("Negative") 
    
    else : 
        print("Neutral") 
  

# sentiment_analysis(score1)
# sentiment_analysis(score2)
# sentiment_analysis(score3)

# print(review3)



# Text Summarization
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize


def create_frequency_table(text_string):
    '''
    Takes in an argument: a text string
    (only use the words that are not part of the stopWords array)
    Returns a dictionary that counts the word frequency from the text.
    '''
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    # process for removing the commoner morphological and inflexional endings from words in English
    ps = PorterStemmer()
    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable

# print(create_frequency_table(review1))
frequency_table = create_frequency_table(review1)


def score_sentences(sentences, freqTable):
    '''
    Takes two arguments:
    the first is sentences (text strings)
    the second is a dictionary (word frequency table)
    Score the sentence using term frequency
    Returns a dictionary 
    '''
    sentenceValue = {}
    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                # first 10 letters 
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]
                    
        # This solves the issue that ong sentences might have an advantage over short sentences.
        sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] // word_count_in_sentence

    return sentenceValue

# print(score_sentences(sent_tokenize(review1),frequency_table))
sentence_value = score_sentences(sent_tokenize(review1),frequency_table)


def find_average_score(sentenceValue):
    '''
    Takes an argument: a dictionary (sentence value)
    Returns an integer: threshold
    '''
    sumValues = 0
    for key in sentenceValue:
        sumValues += sentenceValue[key]

    # Average value of a sentence from original text
    average = int(sumValues / len(sentenceValue))

    return average

# print(find_average_score(sentence_value))
threshold_value = find_average_score(sentence_value)


def generate_summary(sentences, sentenceValue, threshold):
    '''
    Takes three argument: 
    1. sentences: text strings
    2. sentence value: a dictionary that score the sentence using term frequency
    3. threshold: integer (average score)
    Returnns summary of text strings
    (Select a sentence for a summarization if the sentence score is more than the average score)
    '''
    summary = ''

    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] > (threshold):
            summary += " " + sentence
    return summary

# summarization of review 1
print(generate_summary(sent_tokenize(review1), sentence_value, threshold_value))