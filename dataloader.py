import os
from glob import glob

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tqdm import tqdm

import docx
from docx import Document

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
import nlpaug.augmenter.word as naw

import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sentence_splitter import SentenceSplitter, split_text_into_sentences

import warnings
warnings.filterwarnings("ignore")

nltk.download('punkt')
nltk.download("stopwords")
nltk.download('wordnet')


model_name = 'tuner007/pegasus_paraphrase'
DATA_PATH = "data"

doc_structure = {
    'code':[],
    'notes':[],
    'desc':[]
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class preprocessing:

  """
  This function is created for docx files only.
  Required modules:
    - from docx import Document
    - import torch
    - from sentence_splitter import (SentenceSplitter,
                                     split_text_into_sentences
                                    )
    -
  """
  def __init__(self,
               file_name, #make sure that the file is in .docx format
               doc_structure:dict(),
               batch_size=10,
               tokenizer='tuner007/pegasus_paraphrase',
               model='tuner007/pegasus_paraphrase',
               download = True,
               torch_device = device
               ):
    self.file_name = file_name
    self.doc = docx.Document(file_name)
    self.doc_structure = doc_structure
    self.batch_size = batch_size
    self.tokenizer = tokenizer
    self.model = model
    self.download = download
    self.torch_device = torch_device

  def augment_text(self, text):
    # Using synonym augmentation
    aug = naw.SynonymAug(aug_src='wordnet')
    augmented_text = aug.augment(text)
    return augmented_text[0]

  def data_extraction(self, output=False):
    for line in self.doc.paragraphs:
      if line.style.style_id == "Heading2":
        heading=line.text
        splits = heading.split(' ')
        desc = " ".join(splits[1:])
      if line.style.style_id == "Normal":
        self.doc_structure['code'].append(splits[0])
        self.doc_structure['notes'].append(line.text)
        self.doc_structure['desc'].append(desc)

    df = pd.DataFrame(self.doc_structure)
    self.df = df[df.notes.str.contains(' ')==True]
    self.df.to_csv(f"{self.file_name.split('.')[0]}.csv", index=False)

    if output:
      return self.df

  def paraphrasing(self, input_text, num_return_sequences=1, num_beams=10, test=True):
    tokenizer = PegasusTokenizer.from_pretrained(self.model)
    model = PegasusForConditionalGeneration.from_pretrained(self.tokenizer).to(self.torch_device)

    batch = tokenizer([input_text],
                  truncation=True,padding='longest',
                  max_length=50,
                  return_tensors="pt"
                  ).to(self.torch_device)

    translated = model.generate(**batch,
                                max_length=50,
                                num_beams=num_beams,
                                num_return_sequences=num_return_sequences,
                                temperature=1.5)

    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)

    return tgt_text

  def fit(self):
    self.data_extraction()
    splitter = SentenceSplitter(language='en')
    print("Extraction completed!!!")

    self.df["aug_text"] = self.df.notes.apply(self.augment_text)

    # Percentage of the DataFrame to include in each batch
    final_size = int((int(len(self.df)/self.batch_size))*self.batch_size)
    self.df = self.df.sample(frac=1, random_state=23)
    self.df = self.df[:final_size]
    batches = np.split(self.df, self.batch_size)

    bs = 0

    while bs<self.batch_size:
      print(f"Batch {bs + 1}")
      para_df = []
      for idx in tqdm(batches[bs].index, total=len(batches[bs]), desc="paraphrasing"):
        sentence_list = splitter.split(batches[bs]['notes'].loc[idx])
        num_sentence = int(len(sentence_list)*0.55)

        paraphrase = list()
        for i in sentence_list[:num_sentence]:
          para_i = self.paraphrasing(i)
          paraphrase.append(para_i)

        paraphrase = [' '.join(x) for x in paraphrase]
        paraphrase = [' '.join(x for x in paraphrase) ]
        paraphrased_text = str(paraphrase).strip('[]').strip("'")

        para_df.append(paraphrased_text)

      batches[bs]["paraphrase"]=para_df
      batches[bs].to_csv(f"{self.file_name.split('.')[0]}_{bs + 1}.csv", index=False)

      print(f"Batch {bs + 1} Prepared!!!")
      bs+=1

    print("Documents prepared and downloaded to the local system")



def csv_concat(dir):
    all_files = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            all_files.append(os.path.join(root, name))
            
    csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))
    
    return csv_files

csv_files = csv_concat(DATA_PATH)


def file_structuring(csv_files:list()):
    code_file_dict = dict()
    num_file = 0


    for i, f in enumerate(csv_files):
        df = pd.read_csv(f)
        p1, p2, p3 = (df[['notes', 'code', 'desc']], 
                      df[['aug_text', 'code', 'desc']], 
                      df[['paraphrase', 'code', 'desc']])

        p1.columns = ['notes', 'codes', 'desc']
        p2.columns = ['notes', 'codes', 'desc']
        p3.columns = ['notes', 'codes', 'desc']

        df = pd.concat([p1, p2, p3], axis=0, ignore_index=True)
        df = df[['notes', 'codes', 'desc']]
        code_file_dict[i + 1] = df

    df = pd.concat(code_file_dict.values())
    
    return df

