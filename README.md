# MedicalLM
## Classification system for operative notes

In medical coding, an operative note is a documentation that describes the details of a surgical procedure performed on a patient. It provides a comprehensive account of the surgical intervention, including the indications for the procedure, the surgical techniques used, and any findings or complications encountered during the operation.

ICD-10 (International Classification of Diseases, 10th Revision) is a system used for medical coding to classify diseases, injuries, and other health conditions. It is primarily used for coding diagnoses rather than procedures. 

This repo aims to classify hundreds of ICD-10 notes for insurance purposes and even research and development of a well aligned classifier. 

## Dataset
The dataset was not available on the internet which made this project extremely difficult. So the idea was to generate synthetic data from LLMs such as OpenAI's ChatGPT API, Falcon-7b et cetera. 

The dataset was automatically collected and stored on a .docx file. The file had a particular structure where all the classes of the ICD10 codes were stored as "heading2" while the operative notes were collected as "normal" text. This structure made it easy for data engineering and for a creating robust preprocessing pipeline. 

Majority of the data is collected using "prompt engineering" which allowed us collect data on a large scale. But due to the rising cost of generated tokens the dataset was not fully collected. 

For the same reason we partially collected the M13 series. 

### Data engineering and preprocessing

For effective and robust machine learning model we made sure that the dataset doesn't carry unwanted and redundant information. So we took time to maunally access the data as much as possible to find unwanted phrases. Since we generated the data there was a high probability that the phrases might repeat itself. This is one of the major drawback of the LLM. But we also observed there was less to no duplicacy since some of the repeated or similar phrases were hidden in the paragraph pertaining to the operative notes. 

Such phrases don't raise much concerns unless there are the part of the entire paragraph. But other phrases which these LLMs yield as a part of their conversational capabilities could possibly raise concerns. One of such phrases are "Certainly, here are 10 samples...". Apart from that there were other reduntant words which had to be discovered. 

For data exploration we used three major algorithms:
1. Word Cloud  
2. Word Count
3. LDA
 

#### Word Cloud
We used word cloud algorithm to generate visual representations of text data, where the size of each word is proportional to its frequency or importance in the text. The algorithm analyzes the input text, counts the occurrence of each word, and determines how to arrange and size the words in the resulting word cloud.

![Word Cloud](resource/WC.png)


## Model


## Training

## Conclusion
