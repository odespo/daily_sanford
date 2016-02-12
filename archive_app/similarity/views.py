from django.shortcuts import render
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from archive_app.similarity.models import *
from django.template import RequestContext
from django.http import HttpResponseRedirect, HttpResponse
import random
import csv
import datetime
import urllib2
from BeautifulSoup import BeautifulSoup
import xml.etree.ElementTree as ET
import re
import chardet
import string
import archive_app.settings
import scipy.sparse.linalg


##################################################################################### Similarity Calculation + Storage

# Potential Optimizations
# - Stemming
# - Title Factor Optimization
# - Remove Stop Words
# - Thresholding

# Creates tf-idf matrix out of all archived files
def create_tf_idf_mat():
    archived_articles = ArchivedArticle.objects.all()
    
    doc_list = []
    for article in archived_articles:
        total_text = article.content.body + ' '.join([article.content.title]*archive_app.settings.K_SIMILAR_ARTICLES) # Add in article titles
        doc_list.append(total_text)
    
    vectorizer = TfidfVectorizer(min_df=1, smooth_idf = True, stop_words='english')
    arr = vectorizer
    return (vectorizer.fit_transform(doc_list), vectorizer)

# Returns k most similar archived articles in terms of their pks
# Assumes k is less than list of archived articles -> otherwise would be pretty stupid 
# Arbitrarily breaks ties
def find_k_most_similar(target_doc, tf_idf_mat, vectorizer, k):
    # Transform the doc
    transformed_doc = vectorizer.transform([target_doc]).toarray()
    
    l2_norm_doc = None
    
    try:
        l2_norm_doc = scipy.sparse.linalg.norm(transformed_doc)
    except:
        l2_norm_doc = np.linalg.norm(transformed_doc)

    dots = tf_idf_mat.dot(transformed_doc.transpose())
    
    sims = []
    # Calculate cosine similarities
    for i in range(dots.shape[0]):
        cosine_sim = dots[i, 0] / ( l2_norm_doc * scipy.sparse.linalg.norm(tf_idf_mat[i,:]) )
        sims.append((i, cosine_sim))
    
    sorted_ratings = np.sort(np.array(sims, dtype=[("Index", int), ("Rating", float)]), order=["Rating", "Index"], ) # in ascending order, so have to work backwords
    print sorted_ratings
    # Retrieve Best
    best_articles = []
    for i in range(k):
        article_group = sorted_ratings[-1*(i+1)]
        if article_group[1] >= archive_app.settings.SIMILARITY_THRESHOLD: # Needs to be greater than threshold
            best_articles.append(article_group)
    
    return best_articles


# Update/Create similarity relationships for a list of current article pk's
def createSimilarityForPkList(pk_list):
    # First calculate tf-idf matrix.
    tf_idf_mat, vectorizer = create_tf_idf_mat()
    
    for current in CurrentArticle.objects.filter(pk__in=pk_list):
        try:
            # first clear recs from Current Article
            for item in current.archived_articles.all():
                current.archived_articles.remove(item)
            
            whole_text = current.content.body + ' '.join([current.content.title]*archive_app.settings.K_SIMILAR_ARTICLES) # Add in article titles
            # Find and add in new recs
            recs = find_k_most_similar(whole_text,  tf_idf_mat, vectorizer, archive_app.settings.K_SIMILAR_ARTICLES)
            for pk, rating in recs:
                current.archived_articles.add(ArchivedArticle.objects.get(pk=pk))
        except:
            print "There was an error calculating recs for an article" # TODO: add more info

######################################################################################### Archived Model Creation

# Indices of relevant info
id_index = 0
wordid_index = 5
title_index = 1
date_index = 2
author_index = 3
text_index = 4

# Reads in a file where each line is (veridian id\twp_id).
def read_id_file(file_path):
    ids = []
    f = csv.reader(open(file_path, "r"), dialect="excel-tab")
    for contents in f:
        ids.append(contents)
    return ids

# Gets XML data for a given veridian id
def get_veridian_data(ver_id):
    try:
        url = "http://stanforddailyarchive.com/cgi-bin/stanford?a=d&d=" + ver_id + "&f=XML"
        response = urllib2.urlopen(url).read()
        return response
    except:
        print "Could not get XML data for: %s" %(ver_id)
        return None

# Parses a logical section xml to get needed data
def parse_veridian_data(xml_data):
    parsed_data = [None]*5
    root = ET.fromstring(xml_data)
    logical_sec_struct = root[0][0]
    
    for child in logical_sec_struct:
        if child.tag == "LogicalSectionContent":
            for sec_child in child:
                if sec_child.tag == "LogicalSectionTextHTML":
                    html_text = sec_child.text
                    parsed_data[text_index] = BeautifulSoup(html_text, convertEntities=BeautifulSoup.HTML_ENTITIES).contents
        elif child.tag == "LogicalSectionMetadata":
            for sec_child in child:
                if sec_child.tag == "LogicalSectionID":
                        parsed_data[id_index] = sec_child.text
                elif sec_child.tag == "LogicalSectionTitle":
                    title = sec_child.text
                    parsed_data[title_index] = BeautifulSoup(title, convertEntities=BeautifulSoup.HTML_ENTITIES).text
                elif sec_child.tag == "LogicalSectionAuthor":
                    author = sec_child.text
                    author = BeautifulSoup(author, convertEntities=BeautifulSoup.HTML_ENTITIES).text if author != None else None
                    parsed_data[author_index] = author
        elif child.tag == "DocumentMetadata":
            for sec_child in child:
                if sec_child.tag == "DocumentDate":
                    parsed_data[date_index] = sec_child.text

    # Replace None with "None"
    for i,x in enumerate(parsed_data):
        if x == None:
            parsed_data[i] = "None"
    return parsed_data


# Creates an ArchivedArticle instance from a list of attributes to populate it
def create_archived_article(data):
    try:
        new_content = ArticleContent(title=data[title_index])
        body_text = data[text_index]

        # Take out title and author if needed
        body_text = [t.text for t in body_text]
        body_text = body_text[1:] # remove first index of title
        
        if data[author_index] != "None":
            new_content.author = data[author_index]
            author_pos = 0 # Find where the author is and remove that shit
            for i,x in enumerate(body_text):
                if data[author_index] in x:
                    author_pos = i
                    break
            
            body_text = body_text[author_pos+1:]
        
        body_text = ' '.join(body_text)
        new_text = filter(lambda x: x in string.printable, body_text)
        new_content.body = new_text
    
        new_content.save()

        date_info = data[date_index]
        date_info = datetime.datetime.strptime(date_info, '%d %B %Y').strftime('%Y-%m-%d') # Change format so can be put into db

        new_article = ArchivedArticle(veridian_id = data[id_index], content=new_content, wordpress_id=data[wordid_index], published_date=date_info)
        new_article.save()
    except:
        print "There was an error storing article: %s" %(data[0])

# Creates a set of ArchivedArticle from a given file where each line represents (veridian id \t wordpress_id)
def create_archived_from_file(file_path):
    
    # Retrieve Ids
    ids = []
    try:
        ids = read_id_file(file_path)
    except:
        print "there was an error reading in the process and the task must be stopped."
    
    # Parse + structure XML data
    parsed_data = []
    for id_combo in ids:
        try:
            ver_id = id_combo[0]
            wp_id = id_combo[1]
            # Check if already exists first
            if ArchivedArticle.objects.filter(veridian_id=ver_id).exists():
                print "Article with veridian id %s already exists...skipping" %(ver_id)
                continue
            data = get_veridian_data(ver_id)
            if data != None:
                parsed = parse_veridian_data(data)
                parsed.append(wp_id) # Add on wordpress id
                parsed_data.append(parsed)
                       
        except:
            print "there was an error getting and processing the data for article with veridian id %s" %(ver_id)
    # Add to db
    for data in parsed_data:
        create_archived_article(data)


############################################################################################################ Current Article Creation
# TODO: store author
def createCurrentArticles(article_list, include_recs=True): # Takes in list of (word press id, title, author, Published_Date, tags, Text) and adds them to db. Also calculates recommended articles

    pk_list = []
    # Create the current articles
    for wid, title, author, date, tags, text in article_list:
        article = None
        try:
            # First check if already exists
            if CurrentArticle.objects.filter(wordpress_id=wid).exists():
                print "%s already exists...moving on" %(wid)
                continue
            
            # Text processing
            need_text_done = [title, author, text]
            processed_text = []
            for t in need_text_done:
                new_t = filter(lambda x: x in string.printable, t)
                new_t = new_t.replace("\n", " ")
                new_t = new_t.replace("\r", " ")
                new_t = new_t.replace("&euro;&oelig;", "\"")
                new_t = new_t.replace("&euro;&trade;", "\'")
                new_t = new_t.replace("&euro;&tilde;", "\'")
                new_t = new_t.replace("&euro;", "\"")
                processed_text.append(new_t)
            
            title = processed_text[0]
            author = processed_text[1]
            new_text = processed_text[2]




            # Create content
            article_content = ArticleContent(title=title, body=new_text)
            article_content.save()
            
            # Create actual article
            date = datetime.datetime.strptime(date, '%B %d, %Y').strftime('%Y-%m-%d') # Change format so can be put into db

            article = CurrentArticle(published_date=date, content=article_content, wordpress_id=wid)
            article.save()
            pk_list.append(article.pk)
        except:
            print "An error occurred placing %s into the database...Could not be completed" %(wid)
            continue

    # Create recommendations
    if include_recs:
        createSimilarityForPkList(pk_list)

def update_current_article_db(url):
    # Set up connection & get info
    content = None
    try:
        hdr = {'User-Agent': 'Chrome/48.0.2564.103'}
        req = urllib2.Request(url, headers=hdr)
        page = urllib2.urlopen(req)
        content = page.read()
    except:
        print "Could not download the necessary URL...No models added"
        return False
    
    # Parse HTML table to get information
    datasets = []
    try:
        soup = BeautifulSoup(content)
        table = soup.find("table")
        # The first tr contains the field names.
        headings = [td.text for td in table.find("tr").findAll("td")]
        # Get and store the rest: each is a [(column name, val), ...]
        for row in table.findAll("tr")[1:]:
            #dataset = zip(headings, (td.text for td in row.findAll("td")))
            info = [td.text for td in row.findAll("td")]
            datasets.append(info)
    except:
        print "Failed parsing HTML file...No models added"
        return False
    
    # Create Models out of information
    createCurrentArticles(datasets, True)
    

############################################################################################# GET + POST Requests

# TODO: security concerns???? - could ask for API key at end
def getSimilarArchived(request, article_wpId):
    context = RequestContext(request)
    # Find the article we want
    did_find = CurrentArticle.objects.filter(wordpress_id=article_wpId)
    if did_find.count() == 0:
        # Did not find
        return HttpResponse("no id in db") # TODO: maybe change

    article = did_find[0]
    
    recs = article.archived_articles.all()
    rec_ids = []

    for rec in recs:
        rec_ids.append(str(rec.wordpress_id))
    
    if len(rec_ids) == 0:
        # No recs
        return HttpResponse("")
    elif len(rec_ids) == 1:
        # One rec
        return HttpResponse(rec_ids[0])
    else:
        # Multiple recs
        rec_str = ",".join(rec_ids)
        return HttpResponse(rec_str)


def updateCurrentArticles(request):
    url = "http://www.stanforddaily.com/old-posts-recent/" # TODO: probs put into security
    is_good = update_current_article_db(url)
    if not is_good:
        return HttpResponse("failed...check the data and code or try again")
    return HttpResponse("succeeded, but some models may not have been added") # TODO: probably make more sure haha

