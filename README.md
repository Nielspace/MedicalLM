# MedicalLM
## Classification system for operative notes

In medical coding, an operative note is a documentation that describes the details of a surgical procedure performed on a patient. It provides a comprehensive account of the surgical intervention, including the indications for the procedure, the surgical techniques used, and any findings or complications encountered during the operation.

ICD-10 (International Classification of Diseases, 10th Revision) is a system used for medical coding to classify diseases, injuries, and other health conditions. It is primarily used for coding diagnoses rather than procedures. 

This repo aims to classify hundreds of ICD-10 notes for insurance purposes and even research and development of a well aligned classifier. 

## Dataset
The dataset was not available on the internet which made this project extremely difficult. So the idea was to generate synthetic data from LLMs such as OpenAI's ChatGPT API, Falcon-7b et cetera. 

The dataset was automatically collected and stored on a .docx file. The file had a particular structure where all the classes of the ICD10 codes were stored as "heading2" while the operative notes were collected as "normal" text. This structure made it easy for data engineering and for a creating robust preprocessing pipeline. 

Majority of the data is collected using "prompt engineering" which allowed us collect data on a large scale. But due to the rising cost of generated tokens the dataset was not fully collected. 

Although we did collect the M13 series that too partially because of the same reason. 

## Model

## Training

## Conclusion
