#  - common, topic independent key words

# TODO:
# different training data; not news articles
# parts of speech
#   - nouns per sentence
#       - avg and std dev
#   - pronoun
#   - adjectives
#       - avg per noun/pronoun
#       - std dev
#   - verbs
#   - adverbs
#   - preposition
#       - frequency of use compared to nouns
#   - conjunction

import csv
from os import listdir, path
from math import sqrt
import nltk

# a global list since we frequently exclude symbols
# this list should never be modified
symbols = [',', '.', '<', '>', '?', '/', ';', "'", '\\', '|', '{', '}', '[', ']', ':', '"', '-', '=', '_', '+',
               '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '`', '~']

#################################################
# these functions are used to extract syntactical 
# data from sample text

# return the number of syllables in a word
def nsyl(word):
    count = 0
    vowels = 'aeiouy'
    word = word.lower()
    if word[0] in vowels:
        count += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
        if word.endswith('le'):
            count += 1
    if count <= 0:
        count = 1
    return count


# given a message, msg, this function will return
# the average number of syllables per word
def getSylData(word_tokenized_no_sym):
    total = 0
    wordCount = 0
    for word in word_tokenized_no_sym:
        total += nsyl(word)
        wordCount += 1
    return float(total/wordCount)


# given a message, this function returns the standard
# deviation of the number of syllables per word
def getSylStdDev(avg, word_tokenized_no_sym):
    sum = 0
    count = 0
    for word in word_tokenized_no_sym:
        syl = nsyl(word)
        sum += ((syl - avg) ** 2)
        count += 1
    return sqrt((sum/count))


# given a message, this function will return the
# average sentence length and the length of the
# longest sentence in the message
def getSentenceLengthData(msg):
    sentences = msg.split()
    total = 0

    for sent in sentences:
        total += len(nltk.word_tokenize(sent))
    return float(total/len(sentences))


# given a message, this function returns the standard
# deviation of the length of sentences
def getSentenceStdDev(avg, msg):
    sentences = nltk.sent_tokenize(msg)
    sum = 0
    for sent in sentences:
        l = len(sent.split())
        sum += ((l - avg) ** 2)
    return sqrt((sum/len(sentences)))


# given a message, this function will return the
# average number of unique words per 100 words
def getLexicalDiversity(msg):
    words = [word.lower() for word in nltk.word_tokenize(msg)]
    return float((len(set(words))/len(words))*100)


# can probably get rid of this function/data
def getPunctuationData(msg):
    symDict = {sym: msg.count(sym) for sym in symbols}
    totalSymbols = sum(symDict.values())
    return symDict, totalSymbols


# get the average number of commas used per sentence
def comma_avg(sample):
    numComma = 0
    numSent = 0
    sentences = nltk.sent_tokenize(sample)
    for sent in sentences:
        numSent += 1
        for c in sent:
            if c == ',':
                numComma += 1
    return float(numComma/numSent)

# can probably get rid of this function/data
def getSentenceEnders(msg):
    sentences = nltk.sent_tokenize(msg)
    periodCount = 0
    questionCount = 0
    exclimationCount = 0
    total = 0
    for sent in sentences:
        if sent.endswith('.'):
            periodCount += 1
        elif sent.endswith('?'):
            questionCount += 1
        elif sent.endswith('!'):
            exclimationCount += 1
        total += 1
    return float(periodCount/total), float(questionCount/total), float(exclimationCount/total)


#################################################
# these functions are used to extract meaning
# from the sample texts. we use the eight
# different parts of speech & sentence structure
# to accomplish this
#                                                       +------------------------+
#               raw text (string)             +-------> | part of speech tagging |
#                       |                     |         +------------------------+
#                       V                     |                      |
#           +-----------------------+         |                      V
#           | sentence segmentation |         |            pos-tagged sentences
#           +-----------------------+         |          (list of lists of tuples)
#                       |                     |                      |
#                       V                     |                      V
#          sentences (list of strings)        |         +------------------------+
#                       |                     |         |   sentence structure   |
#                       V                     |         |        analyzer        |
#           +-----------------------+         |         +------------------------+
#           |      tokenization     |         |
#           +-----------------------+         |
#                       |                     |
#                       V                     |
#              tokenized sentences            |
#           (list of lists of strings)        |
#                       |                     |
#                       +---------------------+


def preprocessing(sample):  # sample = raw text
    sentences = nltk.sent_tokenize(sample)      # get our list of sentences
    sentences = [nltk.word_tokenize(sent) for sent in sentences]    # get tokenized sentences
    return [nltk.pos_tag(sent) for sent in sentences]

def word_tokenize_no_punctuation(sample):
    tokens = nltk.word_tokenize(sample)
    return [token for token in tokens if token not in symbols]


def pos_frequencies(tagged_sentences):
    pos_list = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS'] # <-- chosen b/c http://www.aicbt.com/authorship-attribution/
    pos_dict = {'NN': 0, 'NNP': 0, 'DT': 0, 'IN': 0, 'JJ': 0, 'NNS': 0}
    word_count = 0
    for sentence in tagged_sentences:
        for word in sentence:
            pos = word[1]
            if pos in symbols:
                continue
            word_count += 1
            if pos in pos_list:
                pos_dict[pos] += 1
    if word_count == 0: return 0
    return [float(pos_dict[key]/word_count) for key in pos_dict]


def adj_adv_percent(pos_tagged_sentences):
    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$']
    adj_tags = ['JJ', 'JJR', 'JJS']
    verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    adv_tags = ['RB', 'RBR', 'RBS', 'RP']
    noun_count = adj_count = verb_count = adv_count = 0
    r1 = r2 = 0
    for sent in pos_tagged_sentences:
        for word in sent:
            if word[1] in noun_tags: noun_count += 1
            elif word[1] in adj_tags: adj_count += 1
            elif word[1] in verb_tags: verb_count += 1
            elif word[1] in adv_tags: adv_count += 1
    if noun_count == 0: r1 = 0
    else: r1 = float(adj_count/noun_count)
    if verb_count == 0: r2 = 0
    else: r2 = float(adj_count/verb_count)
    return [r1, r2]


def avg_conjunctions(pos_tagged_sentences):
    numConj = 0
    for sent in pos_tagged_sentences:
        for tup in sent:
            if tup[1] == 'CC':
                numConj += 1
    return float(numConj/len(pos_tagged_sentences))


def avg_endings(word_tokens):
    count_ing = 0
    count_ed = 0
    for word in word_tokens:
        if word.endswith('ing'):
            count_ing += 1
        elif word.endswith('ed'):
            count_ed += 1
    l = len(word_tokens)
    return [float(count_ing/l), float(count_ed/l)]


# this function will return the "distance" between
# the vocabulary of the sample and the vocabulary
# of a predetermined sample
def chi_square(sample, chi_tokens):
    sample = ([token.lower() for token in sample])
    chi_tokens = ([token.lower() for token in chi_tokens])
    joint_corpus = (sample + chi_tokens)
    joint_freq_dist = nltk.FreqDist(joint_corpus)
    most_common = list(joint_freq_dist.most_common(500))

    # what portion of the joint corpus is made up
    # of the candidate author's tokens
    author_share = (len(sample)/len(joint_corpus))

    # look at 500 most common words in the candidate
    # author's corpus and compare the number of
    # times they can be observed to what would be
    # expected if the author's sample and the
    # disputed sample were both from the same distrobution
    chisquared = 0
    for word, joint_count in most_common:
        # how often do we really see this common word?
        author_count = sample.count(word)
        disputed_count = chi_tokens.count(word)

        # how often should we see it?
        expected_auth_count = joint_count * author_share
        expected_disputed_count = joint_count * (1 - author_share)

        # add the word's contribution to the chi-squared value
        chisquared += (((author_count - expected_auth_count) ** 2)
                       / expected_auth_count)
        chisquared += (((disputed_count - expected_disputed_count) ** 2)
                       / expected_disputed_count)
    return chisquared


def processSample(sample):
    chi_corpus = None
    with open('chi-squared.txt', mode='r') as _file:
        chi_corpus = _file.read()
    chi_tokens = word_tokenize_no_punctuation(chi_corpus)

    training = []
    # preprocess our sample
    tokens = word_tokenize_no_punctuation(sample)
    pos_tagged = preprocessing(sample)
    #################################
    # syntactical analysis
    # get syllable data
    avgSyl = getSylData(tokens)
    training += [avgSyl, getSylStdDev(avgSyl, tokens)]
    # get sentence data
    avgSentLen = getSentenceLengthData(sample)
    training += [avgSentLen, getSentenceStdDev(avgSentLen, sample)]
    # get lexical diversity
    training += [getLexicalDiversity(sample)]
    # get avg num commas per sentence
    training += [comma_avg(sample)]
    # get punctuation data
    period, question, excl = getSentenceEnders(sample)
    training += [period, question, excl]        # <-- up to 9 inputs
    # 100 most common words data
    # ^TODO: maybe?
    #################################
    # "meaning" analysis
    # get pos frequencies of some common parts of speech
    training += pos_frequencies(pos_tagged) # return list of len == 6
    # get adj/noun and adv/verb ratios
    training += adj_adv_percent(pos_tagged)
    # get our chi squared value (how (dis)similar two pieces of text are)
    training += [chi_square(tokens, chi_tokens)]
    # get avg num of CC per sentence
    training += [avg_conjunctions(pos_tagged)]
    training += avg_endings(tokens)
    return training

def generateTraining():
    dir = 'G8'
    dataDirs = [f for f in listdir(dir) if path.isdir('/'.join([dir, f]))]
    outputFiles = ['testing.csv', 'training.csv']

    for dataDir, outputFile in zip(dataDirs, outputFiles):
        numAuthors = len(listdir('/'.join([dir, dataDir])))
        nnOut = [[(1 if j == i else 0) for j in range(numAuthors)] for i in range(numAuthors)]
        with open('/'.join([dir, outputFile]), mode='w', newline='') as _out:
            writer = csv.writer(_out, delimiter=',')
            for author,identifier in zip(listdir('/'.join([dir, dataDir])), nnOut):
                print(author)
                for textFile in listdir('/'.join([dir, dataDir, author])):
                    with open('/'.join([dir, dataDir, author, textFile]), mode='r') as _file:
                        sample = _file.read()
                    training = processSample(sample)
                    # append the output identifiers
                    training += identifier
                    writer.writerow(training)


if __name__ == "__main__":
    if input('Would you like to generate training material? [Y|n] ').lower() == 'y':
        generateTraining()
    else:
        textFile = None
        anonymousSample = None
        while textFile is None:
            textFile = input('What file would you like to de-anonymize? ')
            try:
                with open(textFile, mode='r') as _file:
                    anonymousSample = _file.read()
            except FileNotFoundError:
                print('The file ', textFile, ' does not exist')
                textFile = None
        data = processSample(anonymousSample)
        outFileName = textFile.split('.')[0] + '.csv'
        with open(outFileName, mode='w', newline='') as _out:
            writer = csv.writer(_out, delimiter=',')
            writer.writerow(data)


