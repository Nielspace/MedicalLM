import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

from transformers import pipeline
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas


class text_aug:
    
    """ 
    model-name                 | action     | definition
    ---------------------------|------------------------
    bert-base-uncased          | substitute | Insert word by contextual word embeddings
    ---------------------------|-------------------------
    t5-base                    | summary    | Summarize the text
    ---------------------------|-------------------------
    facebook/bart-large-cnn    | summary    | Summarize the text with a min and max word length
    
    """
    @staticmethod
    def bert_base_uncased(text, action, n_samples):
        aug = naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased', action=action)
        augmented_text = aug.augment(text, n=n_samples)
        return augmented_text
    @staticmethod
    def distilbert_base_uncased(text, action="substitute"):
        aug = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', 
                                        action="substitute")
        augmented_text = aug.augment(text)
        return augmented_text
        
    @staticmethod  
    def summarize_w_t5_base(text):
        aug = nas.AbstSummAug(model_path='t5-base')
        augmented_text = aug.augment(text)
        return augmented_text[0]

    @staticmethod  
    def summarize_w_fb_bart(text, max_length, min_length):
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        out = summarizer(text, max_length=max_length, 
                         min_length=min_length, do_sample=False)
        augmented_text = out["summary_text"]

        return augmented_text







