Code used to build the model...

---
## ```0. Environment Set Up```
---


```python
''' 
run this after restarting runtime to mount google drive directories
'''
from google.colab import drive
drive.mount('/content/drive/') 
```

    Drive already mounted at /content/drive/; to attempt to forcibly remount, call drive.mount("/content/drive/", force_remount=True).



```python
'''
Install the following libraries

'''
# print("installing packages...")
# !pip install wget
# !pip install pyunpack
# !pip install scipy
# !pip install pysub-parser
# !pip install scikit-multilearn
# print("\n done!")
```




    '\nInstall the following libraries\n\n'




```python
'''
Import the following libraries:
'''
print("importing libraries...")
import os
import sys
import ntpath
import subprocess
import zipfile

from bs4 import BeautifulSoup
import requests
import wget

from pysubparser.cleaners import ascii, brackets, formatting, lower_case
from pysubparser import parser

import re
import random

import pandas as pd
import numpy as np
from pyunpack import Archive
import nltk
nltk.download('stopwords')
from scipy import spatial
print("\n done!")
```

    importing libraries...
    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    
     done!



```python
'''
List of paths needed for reusability
'''

notebook_path = "/content/drive/My Drive/Colab Notebooks"

### Transcripts
anime_list_path = f"{notebook_path}/anime_list.txt" # list of anime titles
trans_path = f"{notebook_path}/transcripts/" # where to store the transcripts

### Training and Testing Data
train_path = f"{notebook_path}/train_texts/" # store parsed training data
rdm_train_path = f"{notebook_path}/random_anime/" # store randomized train anime

test_path = f"{notebook_path}/test_texts/" # store parsed testing data
rdm_test_path = f"{notebook_path}/random_anime/" # store randomized test anime
```


```python
''' 
Uncomment this cell to delete everything in a specific folder and its contents. Especially if all folders have been renamed! 
'''

# import shutil
# rm_folder = ["train_texts"]

# print("removing folder(s)...")
# for f in rm_folder:
#   shutil.rmtree(f"{notebook_path}/{f}/")

# print("done!")
```




    ' \n\nUncomment this cell to delete everything in a specific folder and its contents. Especiall if all folders have been renamed! \n'




---
## ```I. Web Scraper```
---

First, we need to build a webscrapper for the kisunekko.net site!

### 1.1. ```Use BeautifulSoup for Webscrapper```


```python
'''
Web-scraper for kistunekko.net
'''

domain = "https://kitsunekko.net"
sub_query = "/dirlist.php?dir=subtitles"
url = domain + sub_query
res = requests.get(url)

res
```




    <Response [200]>




```python
soup = BeautifulSoup(res.content, 'html.parser')

table_res = soup.find(id='flisttable') # id that points to the transcripts
trans_elem = table_res.find_all('a', class_='') # Using the table results, retrieve the rows with links to transcripts
```


```python


''' Strip html tags from text '''
def clean_html(raw_html):
  strip_tags = re.compile('<.*?>')
  clean_text = re.sub(strip_tags, '', raw_html)

  return clean_text
  
```


```python
'''
Each Anime has a title and a link for download
'''
anime_list = {}
for a_tag in trans_elem:
    title_elem = a_tag.find('strong', class_='')
    title = clean_html(str(title_elem))
    anime_list[title] = a_tag["href"]

```


```python
'''Save list of anime for random generator.'''
print("Writing to file...")

with open(anime_list_path, "w+") as f:
  for anime in anime_list.keys():
    f.write(anime)
    f.write("\n")

f.close()

print("Writing complete.")

```

    Writing to file...
    Writing complete.


### 1.2. ```Download compressed files from kisunekko.```


```python
'''
Download file from kisunekko

Dir: where to store the download
URL: link to download the transcripts
'''

def download_files(url, dir):
  zip_path = os.path.expanduser(dir)
  download_to = zip_path + "/"

  
  if not os.path.exists(zip_path):
      os.makedirs(zip_path)
      try:
        wget.download(url=url, out=download_to)

      except:
        pass

```


```python
'''
Get path to zip files for downloads.
'''

print("Downloading from Kistunekko.net...")

for zip_link in anime_list:
  zip_title = zip_link
  zip_url = domain + anime_list[zip_link]
  zip_res = requests.get(zip_url)

  soup = BeautifulSoup(zip_res.content, 'html.parser')

  table_res = soup.find(id='flisttable')
  trans_elem = table_res.find_all('a', class_='')

  for a_tag in trans_elem:
    trans_title = clean_html(str(zip_title))
    download_url = domain + "/" + a_tag["href"]
    download_files(download_url, trans_path + trans_title)

print("Download complete.")
```

#### 1.2.1. Decompress and extract transcripts from zip files within a directory.




```python
'''Check if directory exists. If not, create one. '''
def check_path(file_path):
  if not os.path.exists(file_path):
    print(f"creating dir: {file_path}")
    os.mkdir(file_path)
```


```python
def getListOfFiles(dir_path):

    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dir_path)
    allFiles = list()

    # Iterate over all the entries
    for entry in listOfFile:

        # Create full path
        fullPath = os.path.join(dir_path, entry)

        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles  
```


```python
'''extract files from zip, rar, and .7zip files'''
def decompress_files(trans_folder, trans_path):

  for zip_file in trans_folder:
    if ((".rar" in zip_file) or (".zip" in zip_file) or (".7z" in zip_file)):
      save_to = os.path.splitext(zip_file)[0]
      check_path(save_to)

      with open(zip_file, "rb") as f:
        try:
          Archive(zip_file).extractall(save_to)
          os.remove(zip_file) # remove zip file

        except: # in case of a bad zip file or magic number error
          pass
```


```python
''' Gets a list of directories and subdirectories of the give path'''
def return_path_list(file_path):
  path_list = getListOfFiles(file_path)
  path_list = list()

  for (dirpath, dirnames, filenames) in os.walk(file_path):
      path_list += [os.path.join(dirpath, file) for file in filenames]

  return path_list
```


```python
dir_list = return_path_list(trans_path)

print("Decompressing files...")
decompress_files(dir_list, trans_path)
print("Done!")
```

    Decompressing files...
    Done!


--------
## ```II. Random Generator```
--------

This will be used to randomly select the anime we will train the model on!



### 2.1. ```Write/Read to File```
Write or Read randomized anime list to or from a file


```python
'''write the list of random anime to file'''
def write_rdm_anime(rdm_list, rdm_path, file_name):
  check_path(rdm_path)

  with open(rdm_path + file_name, "w+") as f:
    for anime in rdm_list:
      f.write(anime)
      f.write("\n")

  f.close()
```


```python
'''read the list of random anime to list'''
def read_rdm_anime(rdm_path):
  check_path(rdm_path)

  rdm_list = []
  with open(rdm_path, "r") as f:
    rdm_list = [line.strip() for line in f]

  f.close()

  return rdm_list
```

### 2.2. ```Random Generator```
Generate N out of 2000 anime to train/test the model off of.


```python
''' Read txt file into a list. Select 'N' random anime to be used for training.'''
def list_random(n):
    anime_list = read_rdm_anime(anime_list_path)
  
    random.shuffle(anime_list)
    return anime_list[0:n]
```


```python
rdm_anime_train = list_random(10)
rdm_anime_train
```




    ['Grand Grix no Taka',
     'Gaiking 2005',
     'Concrete Revolutio: Choujin Gensou',
     'Aki Sora',
     'Bobobo-bo Bo-bobo',
     'Dungeon ni Deai wo Motomeru no wa Machigatteiru Darou ka',
     'Akame ga kill',
     'Variable Geo',
     'Radiant S1',
     'Goblin Slayer']




```python
rdm_anime_test = list_random(10)
rdm_anime_test
```




    ['Sailor Moon',
     'Rokushin Gattai GodMars',
     'Major',
     'Tsugumomo',
     'Isekai Quartet',
     'Legend of the Legendaries Heroes',
     'Ikki Tousen Dragon Destiny',
     'Barakamon',
     'Hana no Ko Lun Lun',
     'Maoyuu Maou Yuusha']




```python
print("writing to file...")
write_rdm_anime(rdm_anime_train, rdm_train_path, "/random_train.txt")
write_rdm_anime(rdm_anime_test, rdm_test_path, "/random_test.txt")
print("done!")
```

---
## ```III. Data Preparation```
---
Prep training data for the model using GloVe


### 3.1. ```Clean Transcripts```

After randomly selecting the anime, we will select 10 episodes from each anime and run it through the pysubparser.

#### 3.1.1. File Handling


```python
'''
Save text to text file.
'''
def save_file(text_path, text, save_to):
  check_path(text_path)

  with open(save_to, "w+") as f:
    for line in text:
      f.write(line)

  f.close()
```


```python
''' 
Write transcript text (paragraph) onto a text file. Path depends on if it's to
be used for training data or testing data.
'''
def text_to_file(sub_file, text, text_path):
  base_name = os.path.basename(sub_file)
  file_name = os.path.splitext(base_name)[0]

  save_to = text_path + file_name + ".txt"
  save_file(text_path, text, save_to)
```

#### 3.1.2. Subtitle Parser




```python
'''
Use pysubparser to get parse text from subtitles!

As of Nov 06: 
  Fix:
  (1) extract episode number OR name from file metadata -- let's stay consistent
'''

def parse_subtitle(sub_file, text_path, anime_title):
  text = ''
  text += anime_title + "\n"

  if ".ass" in sub_file or ".srt" in sub_file or ".ssa" in sub_file or ".sub" in sub_file or ".txt" in sub_file:
    subtitles = parser.parse(sub_file)

    # convert subtitltes to lowercase
    lower_sub = brackets.clean(
        lower_case.clean(
            subtitles
        )
    )

    for subtitle in lower_sub:
      text += subtitle.text + " "
      
    text_to_file(sub_file, text, text_path)
```


```python
'''
Given a list of random anime, return all paths and parse the subtitles into training text file.
'''

train_anime_list = read_rdm_anime(rdm_train_path + "/random_train.txt")

print("parsing training subtitles...")
for anime in train_anime_list:
  anime_dir = trans_path + anime
  subtitle_path = return_path_list(anime_dir)

  for sub_file in subtitle_path:
    parse_subtitle(sub_file, train_path, anime)

print("done!")
```

    parsing training subtitles...
    done!


### 3.2. ```Glove Model```

#### 3.2.1. Download GloVe
If not already installed, download here!


```python
''' download the glove file from nlp.stanford'''

glove_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
download_to = '/content/drive/My Drive/datasets/glove/'
```


```python
check_path(download_to)

print("downloading glove dataset")
wget.download(url=glove_url, out=download_to)
```


```python
''' decompress gloVe zip file'''
zip_ref = zipfile.ZipFile(f'{download_to}/glove.6B.zip', 'r')
zip_ref.extractall(f'{download_to}/glove.6B/')
zip_ref.close()
print("done!")
```

#### 3.2.2. Build GloVe Model


```python
'''
Load GloVe model with word embeddings.
'''
def loadGloveModel(file): # from Karishma Malkan on stackoverflow
  print("Loading Glove Model")

  f = open(file,'r', encoding="utf-8") 
  model = {}
  for line in f:
    splitLine = line.split()
    word = splitLine[0]
    embedding = np.array([float(val) for val in splitLine[1:]])
    model[word] = embedding
    
  print("Done.",len(model)," words loaded!") 
  return model
```


```python
'''
Use euclidian distance to find words associated with target word.
'''
def find_similarities(embedding, model):
  return sorted(model.keys(), key=lambda word: spatial.distance.euclidean(model[word], embedding))
```


```python
''' Build GloVe model '''
embed_model = loadGloveModel(f'{download_to}/glove.6B/glove.6B.50d.txt')
```

    Loading Glove Model
    Done. 400000  words loaded!


#### 3.2.3. Labels


```python
''' 
Remove and add words to list of words. Helpful is there are some words you
don't want associated with a label. 
'''
def edit_words(tag, rem_list, add_list):

  for w in rem_list:
    while w in tag: tag.remove(w) 

  for w in add_list:
    if w not in tag:
      tag.append(w)

  # remove duplicates
  tag = list(dict.fromkeys(tag) )
  
  return tag

```


```python
''' 
Use word similarities to find words associated with our labels.
There is also a list of words you can remove or add.
'''
## Romance
romance_list = find_similarities(embed_model['romance'], embed_model)[0:20]
marriage_list = find_similarities(embed_model['divorce'], embed_model)[0:20]
romance = romance_list + marriage_list

rom_rem = ["fantasy", 
           "melodrama", 
           "obsession", 
           "retelling",
           "marriages",
           "revolves",
           "fantasies",
           "explores",
           "mystery"
          "novel",
          "adventures",
           "fascination",
           "fable",
            "heroine",
           "lapsed",
           "romances",
           "divorces",
           "marital",
           "consent"]

rom_add = ["kiss", 
          "lover", 
           "confess", 
           "confession", 
           "engagement", 
           "engaged", 
           "fiance", 
           "fiancee", 
           "boyfriend", 
           "girlfriend"]

romance_tag = edit_words(romance, rom_rem, rom_add)
```


```python
## Supernatural
magic_list = find_similarities(embed_model['magical'], embed_model)[0:20]
creatures_list = find_similarities(embed_model['vampire'], embed_model)[0:20]
supernatural = magic_list + creatures_list

super_rem = ["marvelous", 
             "wondrous", 
             "cinematic", 
             "imagination", 
             "essence", 
             "protagonist", 
             "villain", 
             "rabbit", 
             "spider", 
             "inspiration", 
             "fantastic", 
             "sorts"]

super_add = ["psychic", 
             "cursed", 
             "spirit", 
             "ghost", 
             "haunted", 
             "zombie", 
             "demon", 
             "monk"]

supernatural_tag = edit_words(supernatural, super_rem, super_add)
```


```python
## Death
death_list = find_similarities(embed_model['death'], embed_model)[0:20]
murder_list = find_similarities(embed_model['murder'], embed_model)[0:20]
death = death_list + murder_list

death_rem = ["taken", 
             "another", 
             "brought", 
             "father", 
             "was", 
             "birth"]

death_add = ["funeral", 
             "criminal", 
             "arrest", 
             "abduction"]

death_tag = edit_words(death, death_rem, death_add)
```


```python
# length of label vectors used are the same
print(f"romance: {len(romance_tag)} \nsupernatural: {len(supernatural_tag)} \ndeath: {len(death_tag)}")
```

    romance: 33 
    supernatural: 33 
    death: 33



```python
'''
If an word from one of the below label vectors (death, supernatural, romance) 
is found in unique_words[], tag it.

Nov 7

Suggestion:
(1) Can we look for variations of a word by stemming the vectors?
'''
def check_death(unique_words):
  for w in unique_words:
    if w in death_tag:
      return True
      break

def check_supernatural(unique_words):
  for w in unique_words:
    if w in supernatural_tag:
      return True
      break

def check_romance(unique_words):
  for w in unique_words:
    if w in romance_tag:
      return True
      break

```

#### 3.2.4. Transcript Normalization


```python
'''Normalize transcript '''
def normalize_document(doc):
    wpt = nltk.WordPunctTokenizer()
    stop_words = nltk.corpus.stopwords.words('english')
    
    # lower case and remove special characters and whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()

    # tokenize document
    tokens = wpt.tokenize(doc)

    # filter out stopwords from document
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    
    return doc
```


```python
'''
Read text from a file to a list.
'''
def read_text(text_file):
  transcript = []
  title = ""
  base_name = os.path.basename(text_file)
  episode_name = os.path.splitext(base_name)[0]
  
  with open(text_file, mode="r", encoding="utf-8") as f:
    title = f.readline()

    for line in f:
      c_line = re.sub(r'\{(.*?)\}', '', line, re.I|re.A)
      c_line = c_line.lower()
      c_line = c_line.strip()
      transcript.append(c_line.strip())


  return np.array(transcript), title, episode_name  
```


```python
'''
For each text file in the training path, save the transcript, anime information,
and labels to later be used for the model.
'''

text_list = return_path_list(train_path)

pd_trans = [] # transcripts
pd_title = [] # anime title
pd_epname = [] # episode name

pd_rom = [] # romance
pd_supernat = [] # supernatural
pd_death = [] # death

''' Read content from each text file and categorize it '''
print("reading to file....")
for text_file in text_list:
  transcript, title, epname = read_text(text_file)
  pd_trans.append(transcript)
  pd_title.append(title.strip().split('\n'))
  pd_epname.append(epname)

  # normalize transcript
  vec_transcript = np.vectorize(normalize_document)
  norm_transcript = vec_transcript(transcript)

  # get unique words from the transcript
  unique_words = list(set([word for sublist in [trans.split() for trans in norm_transcript] for word in sublist]))

  # check if category appears in the transcript
  has_death = check_death(unique_words)
  pd_death.append(has_death)

  has_supernat = check_supernatural(unique_words)
  pd_supernat.append(has_supernat)

  has_rom = check_romance(unique_words)
  pd_rom.append(has_rom)

print("done!")
```

    reading to file....
    done!


### 3.3. ```Convert Dataset to DataFrame```


```python
''' 
Convert dataset to dataframe for further manipulation and better visualization.

Nov 7
Fix:
(1) replace episode name with episode # (if necessary)

 '''
def create_training_df(ep_name, anime_title, anime_transcript, death_label, supernat_label, rom_label):
    df = pd.DataFrame({'text': anime_transcript, 
                          'anime_title': anime_title,
                          'death': death_label,
                          'supernatural': supernat_label,
                          'romance': rom_label,
                          'episode_title': ep_name})

    anime_df = df[['text', 'anime_title', 'death','supernatural', 'romance', "episode_title"]]

    df = convert_labels(anime_df)

    return df
```


```python
''' Replace True with 1 and False with 0 '''
def convert_labels(df):
  df['death'] = [1 if x == True else 0 for x in df['death']]
  df['supernatural'] = [1 if x == True else 0 for x in df['supernatural']]
  df['romance'] = [1 if x == True else 0 for x in df['romance']]

  return df

```


```python
'''
Create dataframe for training data and save as csv.
'''

anime_df = create_training_df(pd_epname, pd_title, pd_trans, pd_death, pd_supernat, pd_rom) 
anime_df.to_csv(f"{notebook_path}/train_anime.csv", index=False)
anime_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>anime_title</th>
      <th>death</th>
      <th>supernatural</th>
      <th>romance</th>
      <th>episode_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[gaiking gai gai gai daiku maryu gaiking gai g...</td>
      <td>[Gaiking 2005]</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Gaiking Legend of Daiku-Maryu - 39_track3_eng</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[gaiking gaiking gai gai gai daiku maryu gaiki...</td>
      <td>[Gaiking 2005]</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Gaiking Legend of Daiku-Maryu - 01_track3_eng</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[gaiking gaiking gai gai gai daiku maryu gaiki...</td>
      <td>[Gaiking 2005]</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Gaiking Legend of Daiku-Maryu - 02_track3_eng</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[gaiking gaiking gai gai gai daiku maryu gaiki...</td>
      <td>[Gaiking 2005]</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Gaiking Legend of Daiku-Maryu - 03_track3_eng</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[gaiking gaiking gai gai gai daiku maryu gaiki...</td>
      <td>[Gaiking 2005]</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Gaiking Legend of Daiku-Maryu - 04_track3_eng</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>120</th>
      <td>[master, what should i do today? what should y...</td>
      <td>[Goblin Slayer]</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>[Moozzi2] Goblin Slayer - 08 (BD 1920x1080 x.2...</td>
    </tr>
    <tr>
      <th>121</th>
      <td>[there and back again now, then... what's rath...</td>
      <td>[Goblin Slayer]</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>[Moozzi2] Goblin Slayer - 09 (BD 1920x1080 x.2...</td>
    </tr>
    <tr>
      <th>122</th>
      <td>[when i was a child, i thought i'd become an a...</td>
      <td>[Goblin Slayer]</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>[Moozzi2] Goblin Slayer - 10 (BD 1920x1080 x.2...</td>
    </tr>
    <tr>
      <th>123</th>
      <td>[a gathering of adventurers good morning! just...</td>
      <td>[Goblin Slayer]</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>[Moozzi2] Goblin Slayer - 11 (BD 1920x1080 x.2...</td>
    </tr>
    <tr>
      <th>124</th>
      <td>[how did it come to this? that horde is finish...</td>
      <td>[Goblin Slayer]</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>[Moozzi2] Goblin Slayer - 12 END (BD 1920x1080...</td>
    </tr>
  </tbody>
</table>
<p>125 rows × 6 columns</p>
</div>



---
## ```IV. Build Model```
---

The model we will be using is K-Nearest Neighbors.

[explaination why here]


```python
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer 

# Models
from skmultilearn.adapt import MLkNN # K-Nearest Neighbors
from skmultilearn.problem_transform import BinaryRelevance # Binary Relevance
from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bayes
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score 
```

### 4.1. ```Train Model```


```python
'''
Read from CSV and vectorize dataset for training the model.
'''

anime_df = pd.read_csv(f'{notebook_path}/train_anime.csv') 
X = anime_df["text"] 
y = np.asarray(anime_df[["death","supernatural", "romance"]]) 
  
# initializing TfidfVectorizer  
vect_tf = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,2), max_df=0.85)

# fitting the tf-idf on the given data 
vect_tf.fit(X)
```




    TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',
                    dtype=<class 'numpy.float64'>, encoding='utf-8',
                    input='content', lowercase=True, max_df=0.85, max_features=None,
                    min_df=1, ngram_range=(1, 2), norm='l2', preprocessor=None,
                    smooth_idf=True, stop_words=None, strip_accents='unicode',
                    sublinear_tf=False, token_pattern='(?u)\\b\\w\\w+\\b',
                    tokenizer=None, use_idf=True, vocabulary=None)




```python
'''
Train model using vectorized datasets.
'''

# split the data into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, shuffle=True) 
  
# transform datasets
vect_Xtrain = vect_tf.transform(X_train) 
vect_Xtest = vect_tf.transform(X_test)
```

#### 4.1.1. Compare Models
We will choose the best model to use by comparing the following models' accuracy.

1. MLkNN Model
2. Naiive Bayes Model


```python
threshold = 0.3 # threshold value
```


```python
'''
Use MLkNN model for multilabel classification
'''
mlknn_classifier = MLkNN()
mlknn_classifier.fit(vect_Xtrain, y_train)

y_pred = mlknn_classifier.predict(vect_Xtest)
  
# accuracy
y_thres = (y_pred >= threshold).astype(int)

print(f"K-Nearest Neighbors Model \n ------------")
print(f"f1-score: {f1_score(y_test, y_pred, average='micro')}")
print(f"f1-score threshold: {f1_score(y_test, y_thres, average='micro')}")
```

    K-Nearest Neighbors Model 
     ------------
    f1-score: 0.9
    f1-score threshold: 0.9



```python
'''
Use Naive Bayes for multilabel classification
'''

# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
bayes = BinaryRelevance(GaussianNB())

# train
bayes.fit(vect_Xtrain, y_train)

# predict
y_pred = bayes.predict(vect_Xtest) 

# accuracy
y_thres = (y_pred >= threshold).astype(int) # threshold

print(f"Naiive Bayes Model \n ------------")
print(f"f1-score: {f1_score(y_test, y_pred, average='micro')}")
print(f"f1-score threshold: {f1_score(y_test, y_thres, average='micro')}")
```

    Naiive Bayes Model 
     ------------
    f1-score: 0.8734177215189874
    f1-score threshold: 0.8734177215189874


### 4.2. ```Test Model Predictions```
Whatever we do to the training data, we do the same to the testing data!


__Nov 6 Note__: We need more data to improve the model!


```python
'''
Given a list of random anime, return all paths and parse the subtitles into test text file.
'''

test_anime_list = read_rdm_anime(rdm_test_path + "/random_test.txt")

print("\nparsing test subtitles...")
for anime in test_anime_list:
  anime_dir = trans_path + anime
  subtitle_path = return_path_list(anime_dir)

  for sub_file in subtitle_path:
    parse_subtitle(sub_file, test_path, anime)

print("done!")
```

    
    parsing test subtitles...
    done!



```python
# glimpse at the list of anime
test_anime_list
```




    ['Sailor Moon',
     'Rokushin Gattai GodMars',
     'Major',
     'Tsugumomo',
     'Isekai Quartet',
     'Legend of the Legendaries Heroes',
     'Ikki Tousen Dragon Destiny',
     'Barakamon',
     'Hana no Ko Lun Lun',
     'Maoyuu Maou Yuusha']




```python
'''
Get path of test data texts and append each text to a list of transcripts.
'''
test_list = return_path_list(test_path)

pd_test_trans = []
pd_test_title = []
pd_test_epname = []

''' Read content from each text file and categorize it '''
for text_file in test_list:
  transcript, title, epname = read_text(text_file)
  pd_test_trans.append(transcript)
  pd_test_title.append(title.strip().split('\n'))
  pd_test_epname.append(epname)
```


```python
'''
Convert list of texts to dataframe, along with episode name and labels.
'''
df = pd.DataFrame({'text': pd_test_trans,
                        'anime_title': pd_test_title,
                   'episode_title': pd_test_epname})

test_df = df[['text', 'anime_title', 'episode_title']]

text_df = test_df[['text']]
vect_test = vect_tf.transform(text_df) 
  
pred_labels = mlknn_classifier.predict(vect_test)
pred_labels = pred_labels.toarray()

pred_labels
```




    array([[1, 1, 0]])




```python
'''
save preditions to dataframe and print results
'''
pred_death = [row[0] for row in pred_labels]
pred_supernat = [row[1] for row in pred_labels]
pred_romance = [row[2] for row in pred_labels]

label_df = pd.DataFrame({"Death": pred_death,
                        "Supernatural": pred_supernat,
                        "Romance": pred_romance})

# join anime information with their predicted labels
pred_df = pd.concat([test_df, label_df], axis=1)

pred_df.to_csv(f"{notebook_path}/test_anime.csv", index=False)
pred_csv
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>anime_title</th>
      <th>episode_title</th>
      <th>Death</th>
      <th>Supernatural</th>
      <th>Romance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>['subtitles by explosiveskull www.opensubtitle...</td>
      <td>['Major']</td>
      <td>Batman.Death.in.the.Family.2020.REPACK.720p.Bl...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>['when the obi, wound up like a cocoon, unfurl...</td>
      <td>['Tsugumomo']</td>
      <td>[Moozzi2] Tsugumomo - 01 (BD 1920x1080 x.264 F...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>['kazuya! come, it is morning. honestly, sleep...</td>
      <td>['Tsugumomo']</td>
      <td>[Moozzi2] Tsugumomo - 02 (BD 1920x1080 x.264 F...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>['you there. th... they\'re huge! you are kaga...</td>
      <td>['Tsugumomo']</td>
      <td>[Moozzi2] Tsugumomo - 03 (BD 1920x1080 x.264 F...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>['water impact. oh, no. oho. how\'s that, kuku...</td>
      <td>['Tsugumomo']</td>
      <td>[Moozzi2] Tsugumomo - 04 (BD 1920x1080 x.264 F...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>69</th>
      <td>["i-i... i-i've brought you some tea! thank yo...</td>
      <td>['Maoyuu Maou Yuusha']</td>
      <td>[Moozzi2] Maoyuu Maou Yuusha - 08 (BD 1920x108...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>70</th>
      <td>['the central nations have made their move, it...</td>
      <td>['Maoyuu Maou Yuusha']</td>
      <td>[Moozzi2] Maoyuu Maou Yuusha - 09 (BD 1920x108...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>71</th>
      <td>['to summarize, by accepting her speech, we\'v...</td>
      <td>['Maoyuu Maou Yuusha']</td>
      <td>[Moozzi2] Maoyuu Maou Yuusha - 10 (BD 1920x108...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>72</th>
      <td>['there\'s a 3 point increase! buy now, as muc...</td>
      <td>['Maoyuu Maou Yuusha']</td>
      <td>[Moozzi2] Maoyuu Maou Yuusha - 11 (BD 1920x108...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>73</th>
      <td>['mage! so sleepy... this is one of the legend...</td>
      <td>['Maoyuu Maou Yuusha']</td>
      <td>[Moozzi2] Maoyuu Maou Yuusha - 12 END (BD 1920...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>74 rows × 6 columns</p>
</div>



--End--
