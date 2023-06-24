import json
import pandas as pd
import os

def iter_corpus_file(filename_corpus):
    with open(filename_corpus, 'rt') as corpusfile:
        for l in corpusfile:
            doc = dict()
            json_file = json.loads(l)
            doc['id'] = json_file["_id"]
            # doc['contents'] = json_file["title"]+ '. ' + json_file["text"] # for the nq
            doc['contents'] = json_file["text"] # for hotpotqa
            doc['title'] = json_file["title"]
            yield doc
            
             
def iter_queries_file(filename_queries):
    with open(filename_queries, 'rt') as queries_file:
        for l in queries_file:
            doc = dict()
            json_file = json.loads(l)
            doc['id'] = json_file["_id"]
            doc['text'] = json_file["text"]
            yield doc
            

data_folder = 'hotpotqa'

filename_corpus = f'{data_folder}/corpus.jsonl'
docs = [x for x in iter_corpus_file(filename_corpus)]

os.makedirs(f'{data_folder}_serini_jsonl', exist_ok=True)
with open(f'{data_folder}_serini_jsonl/corpus.jsonl', 'w') as outfile:
    for entry in docs:
        json.dump(entry, outfile)
        outfile.write('\n')
        
filename_queries = f'{data_folder}/queries.jsonl'
queries = [x for x in iter_queries_file(filename_queries)]


qrel_df = pd.read_csv(f'{data_folder}/qrels/test.tsv', sep='\t')

queries_df = pd.DataFrame(queries)
queries_df.to_csv(f'{data_folder}/all_queries.tsv', sep='\t', header=False, index=None) 

test_queries_df = queries_df[queries_df['id'].isin(qrel_df['query-id'])]
test_queries_df.to_csv(f'{data_folder}/queries.tsv', sep='\t', header=False, index=None) 

if data_folder == 'hotpotqa':
    test_ids = set(qrel_df['query-id'])
    test_queries = [q for q in queries if q['id'] in test_ids]
    
    with open(filename_queries, 'w') as f:
        for q in test_queries:
            q['_id'] = q['id']
            _ = q.pop('id')
            json.dump(q, f)
            f.write('\n')