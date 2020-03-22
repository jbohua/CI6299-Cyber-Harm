import os
import re
import copy
import csv
import pdfminer.high_level
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer

references_directory = './Medical References'
taxonomy_match_directory = './Taxonomy-Match'
taxonomy_file_path = './taxonomy.csv'
sentence_context_length = 1

taxonomy_headers = []
taxonomy_rows = []
default_keywords_matched_sentences = {}

def clean_text(input):
    
    for linebreak in ['\n-\n', '\n-', '-\n\n', '-\n']:

        input = input.replace(linebreak, '')

    input = re.sub(r'\n+', ' ', input)

    input = re.sub(r'\s+', ' ', input)

    return input

def format_string(input):

    return input.strip().lower()

def validate_directory(path):

    if not os.path.exists(path):
        os.makedirs(path)

def initialize_taxonomys():
    
    global taxonomy_headers
    global taxonomy_rows
    global default_keywords_matched_sentences

    with open(taxonomy_file_path, 'r', encoding='utf8') as csvfile:
        
        csvreader = csv.reader(csvfile)

        taxonomy_headers = next(csvreader)
        
        for taxonomy_row in csvreader:
            
            taxonomy_rows.append(taxonomy_row)

            element_name = format_string(taxonomy_row[2])
            
            for synonym in taxonomy_row[3].split(','):

                formatted_synonym = format_string(synonym)
                
                if formatted_synonym not in default_keywords_matched_sentences: 

                    default_keywords_matched_sentences[formatted_synonym] = []

def match_taxonomys(keywords_matched_sentences, keywords_matached_csv):
    
    has_any_matched_key_word = False

    with open(keywords_matached_csv, 'w', encoding='utf8') as csvfile:
        
        taxonomy_match_file_writer = csv.writer(csvfile)

        taxonomy_match_file_writer.writerow(taxonomy_headers)

        for taxonomy_row in taxonomy_rows:

            has_matched_key_word = False

            keyword_macthes_row = copy.deepcopy(taxonomy_row)

            synonyms = taxonomy_row[3]
            
            for synonym in synonyms.split(','):
                
                formatted_synonym = format_string(synonym)

                if len(keywords_matched_sentences[formatted_synonym]) > 0:

                    has_matched_key_word = True

                    keyword_macthes_row += keywords_matched_sentences[formatted_synonym]

            if has_matched_key_word is True:

                has_any_matched_key_word = True

                taxonomy_match_file_writer.writerow(keyword_macthes_row)

    if has_any_matched_key_word is False:

        os.remove(keywords_matached_csv)

def get_sentence_context(sentences, current_index):
    
    long_sentence = ''

    for i in range(current_index - sentence_context_length, current_index + sentence_context_length + 1): 
        
        if i >=0 and i < len(sentences):
            long_sentence += sentences[i] + ' '
    
    return long_sentence

def process_reference(original_reference, keywords_matached_csv):
    
    original_fulltext = pdfminer.high_level.extract_text(original_reference)
    
    fulltext = clean_text(original_fulltext)
    
    keywords_matched_sentences = copy.deepcopy(default_keywords_matched_sentences)
    
    sentences = sent_tokenize(fulltext)

    for i in range(0, len(sentences)):

        sentence = sentences[i]

        long_sentence = get_sentence_context(sentences, i)

        lowered_sentence = format_string(sentence)
        
        features = {}

        for keyword in keywords_matched_sentences:
            
            ngram_range = len(keyword.split(' '))

            if ngram_range not in features:

                try:
                    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None,analyzer='word', ngram_range=(ngram_range, ngram_range))
                
                    vectorizer.fit([lowered_sentence])
                
                    features[ngram_range] = vectorizer.vocabulary_
                
                except ValueError as e:
                    print(e)
                    continue

            if keyword in features[ngram_range]:

                keywords_matched_sentences[keyword].append(long_sentence)
    
    match_taxonomys(keywords_matched_sentences, keywords_matached_csv)

def process_references():

    for reference_file in os.listdir(references_directory):

        filename, file_extension = os.path.splitext(reference_file)

        if file_extension.lower() == '.pdf':
            process_reference(references_directory + '/' + reference_file, taxonomy_match_directory + '/' + reference_file.split('.')[0] + '.csv')

if __name__ == "__main__":
    
    validate_directory(references_directory)

    validate_directory(taxonomy_match_directory)
    
    initialize_taxonomys()

    process_references()