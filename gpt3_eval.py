import os
import inspect
import openai
import numpy as np
import random
import matplotlib.pyplot as plt

from nltk.corpus import wordnet
from time import sleep

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
os.chdir(path)

from words_ressources import words_list

## init openai with key and organization

openai.organization = "Your_org_key_here"
openai.api_key = "your_personnal_key_here"
#keys can be obtained from your account on openai's website

#retrieve engines in order to use them
openai.Engine.retrieve("ada")
openai.Engine.retrieve("babbage")
openai.Engine.retrieve("curie")

## functions

def deduplicate_list(list):
    '''
    Removes any duplicated items from a list
    '''
    return np.unique(np.array(list)).tolist()

def countCommonWords(result, list):
    '''
    Counts the number of common words between the ouput of the NLP system and a reference list
    <result> is a single string in which words must be separated by a comma (',')
    Identical words are eliminated, spaces are deleted
    '''

    result = result.replace(" ", "") #remove spaces

    result = deduplicate_list(result.split(',')) #turn string into list and deduplicate items

    cnt = 0
    for wrd in result:
        if wrd in list:
            cnt += 1

    return cnt

def generateSynonyms(wrd_list):
    '''
    Uses the wordnet database to find synonyms of a given word.
    A list of non duplicated lists of synonyms is returned
    '''

    synonyms_list = []

    for word in wrd_list:
        synonyms = []

        for syn in wordnet.synsets(word):
            for lm in syn.lemmas():
                if lm.name() != word.lower():
                    synonyms.append(lm.name())

        synonyms_list.append(deduplicate_list(synonyms))

    return synonyms_list

def createPrompt(wrds, syns):
    '''
    Creates a prompt to be interpreted by an openai completion engine from a list of examples and a subject to be completed by the engine
    The syninyms are not written for last item of <wrds>, and must be guessed by the system
    '''
    prompt = "The following is a list of words and their synonyms\n\n"

    for i, wrd in enumerate(wrds[:len(wrds)-1]):
        if len(syns[i])>=2: #make sure the word has at least two synonyms
            prompt = prompt + wrd + ": " #add example words
            for syn in random.sample(syns[i], random.randint(2,len(syns[i]))):
                prompt = prompt + syn + ', ' #add example synonyms

            prompt = prompt[:len(prompt)-2] + '\n' #remove the last two characters (', ') that are not needed

    prompt = prompt + wrds[len(wrds)-1] + ":" #add the last word with no synonyms

    return prompt


## main

def main()
    #scores containers
    ada_score_list = []
    babbage_score_list = []
    curie_score_list = []
    ex_num_list = []

    for examples_num in [0, 1, 2, 3, 5, 7, 9, 11, 13, 15]:
        print('evaluation engines with ' + examples_num + ' examples.')

        #scores for the current loop
        ada_score = 0
        babbage_score = 0
        curie_score = 0

        for t in range(10):
            for i in range(10):
                #create a prompt from random words from the filtered list
                sampled_words = random.sample(words_list, examples_num+1)
                sampled_synonyms = generateSynonyms(sampled_words)
                prmpt = createPrompt(sampled_words, sampled_synonyms)

                #call each engine and keep track of the score
                ada_response = openai.Completion.create(
                engine="ada",
                prompt= prmpt,
                max_tokens=20,
                temperature=0,
                top_p = 1.0,
                frequency_penalty = 0.8,
                stop = ["\n", "."]
                )

                ada_score += countCommonWords(ada_response["choices"][0]["text"], sampled_synonyms[len(sampled_synonyms)-1])

                babbage_response = openai.Completion.create(
                engine="babbage",
                prompt= prmpt,
                max_tokens=20,
                temperature=0,
                top_p = 1.0,
                frequency_penalty = 0.8,
                stop = ["\n", "."]
                )

                babbage_score += countCommonWords(babbage_response["choices"][0]["text"], sampled_synonyms[len(sampled_synonyms)-1])

                curie_response = openai.Completion.create(
                engine="curie",
                prompt= prmpt,
                max_tokens=20,
                temperature=0,
                top_p = 1.0,
                frequency_penalty = 0.8,
                stop = ["\n", "."]
                )

                curie_score += countCommonWords(curie_response["choices"][0]["text"], sampled_synonyms[len(sampled_synonyms)-1])
            #wait for 60 seconds every 10 loops in order to stay below the maximum number of requests per minute
            sleep(60)

        #fill containers with the scores from the current loop
        ada_score_list.append(ada_score)
        babbage_score_list.append(babbage_score)
        curie_score_list.append(curie_score)
        ex_num_list.append(examples_num)

    #plot results
    plt.plot(ex_num_list, ada_score_list, '.-')
    plt.plot(ex_num_list, babbage_score_list, '.-')
    plt.plot(ex_num_list, curie_score_list, '.-')

    plt.xlabel('Number of examples')
    plt.ylabel('Score')

    plt.legend(("ada","babbage", "curie"))

    plt.show()

    return 0

if __name__ == "__main__":
    main()