from transformers import pipeline
from pythainlp.tokenize import sent_tokenize
from bs4 import BeautifulSoup
import requests
import re
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import concurrent.futures
from collections import defaultdict
from functools import partial

import streamlit as st

st.title('Uber pickups in NYC')


# Cliam Detection
claim_detection = pipeline("text-classification", model="Nithiwat/mdeberta-v3-base_claim-detection")

# Evidence Retrieval
## Document Search

def search(query):
  try:
      from googlesearch import search
  except ImportError:
      print("No module named 'google' found")
 
  claim = query
  sites = ['wikipedia.org','factcheckthailand.afp.com','www.antifakenewscenter.com/','www.factcheck.org/','https://www.bbc.com/thai']
  query = claim + ' OR '.join(['site:'+x for x in sites])
  link = []
  for j in search(query, tld="co.th", num=10, stop=10, pause=2, lang='th'):
      link += [j]
  return link


def get_text(url):
  URL = url
  page = requests.get(URL)
  soup = BeautifulSoup(page.content, "html.parser")
  tag = soup.body
  text = ""
  for para in tag.find_all("p"):
    text += para.get_text(strip=True)+" "
  return text

def get_evidence(link):
  all_text=''
  try:
    all_text = get_text(link)
  except Exception as e: 
    all_text = str(e)
    pass

  all_text = all_text.replace(u'\xa0', ' ')
  all_text = all_text.replace('เนื้อหาmove to sidebarซ่อน', '')
  all_text = all_text.replace('หน้าสำหรับผู้แก้ไขที่ออกจากระบบเรียนรู้เพิ่มเติม', '')
  all_text = re.sub('\[\d+\]', ' ', all_text)
  sentences = sent_tokenize(all_text, engine="crfcut")
  return sentences

## Sentence Ranker
encoder_name = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1" #@param ["cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "cross-encoder/mmarco-mdeberta-v3-base-5negs-v1"]
encoder = CrossEncoder(encoder_name)

def rank_evidence(claim, sentences):
  queries = [claim] 
  passages = sentences
  for query in queries:
    model_inputs = [[query, passage] for passage in passages]
    scores = encoder.predict(model_inputs)

    results = [{'input': inp, 'score': score} for inp, score in zip(model_inputs, scores)]
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    return results

# Verdict Prediction
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

def verdict(claim, evidence):
  input = tokenizer(claim, evidence, truncation=True, return_tensors="pt")
  output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
  prediction = torch.softmax(output["logits"][0], -1).tolist()
  label_names = ["entailment", "neutral", "contradiction"]
  prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
  prediction = dict(sorted(prediction.items(), key=lambda item: item[1]))
  return [list(prediction.keys())[-1], prediction[list(prediction.keys())[-1]]]


####### Put it all together #######

text_input = '''สถาบันทางวิทยาศาตร์อนามัยนี่คือยาที่มีผลต่อมะเร็งดีเยี่ยมล่าสุดของโลก มะนาวเป็นผลิตผลที่มหัศจรรย์มากที่สามารถฆ่าเซลมะเร็งได้มากกว่า 1 หมื่นเท่า'''

# Claim Detection
text_input = sent_tokenize(text_input, engine="crfcut")
check_worthy = claim_detection(text_input)
all_claim = list(zip(text_input, check_worthy))
check_worthy_claim = []
for claim in all_claim:
  if(claim[1]['label'] == 'LABEL_1'):
    check_worthy_claim += [claim[0]]

# Evidence Retrieval
def get_parallel_link(claim, res):
    link = search(claim)
    json_list = [{'claim': claim, 'link': i} for i in link]
    res.extend(json_list)

MAX_THREADS = 20

def request_multithread(urls, scrape_func):
    threads = min(MAX_THREADS, len(urls))    
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(scrape_func, urls)

res = []

request_multithread(check_worthy_claim, partial(get_parallel_link, res=res))

def get_document(url, res):
    sentences = get_evidence(url['link'])
    json_sentences = [{url['claim']:sentences}]
    res.extend(json_sentences)

    keys = sentences
    values = [url['link'] for i in sentences]
    sources.update(dict(zip(keys, values)))

MAX_THREADS = 20

def scrape_multithread(urls, scrape_func):
    threads = min(MAX_THREADS, len(urls))    
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(scrape_func, urls)

document = []
sources = {}
scrape_multithread(res, partial(get_document, res=document))

result = defaultdict(list)
 
for i in range(len(document)):
    current = document[i]
    for key, value in current.items():
        for j in range(len(value)):
            result[key].append(value[j])
 
all_evidence = dict(result)

for evidence in all_evidence:
  evidence_list = rank_evidence(evidence, all_evidence[evidence])
  evidence_list = [i['input'][1] for i in evidence_list][:10]
  all_evidence[evidence] = evidence_list

# Verdict Prediction
for claim in all_evidence:
  output = []
  for evidence in all_evidence[claim]:
    prediction = verdict(claim, evidence)
    if prediction[0] != 'neutral':
      output.append([evidence,prediction,sources[evidence]])
  all_evidence[claim] = output

print(all_evidence)

