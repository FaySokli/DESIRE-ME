import pandas as pd

#### QUERIES ####
with open('/home/ubuntu/esokli/DESIRE-ME/nfcorpus/train.all.queries', 'r') as file:
    qtrain = file.read()
with open('/home/ubuntu/esokli/DESIRE-ME/nfcorpus/test.all.queries', 'r') as file:
    qtest = file.read()
with open('/home/ubuntu/esokli/DESIRE-ME/nfcorpus/dev.all.queries', 'r') as file:
    qdev = file.read()

qtrain_s = qtrain.split('\n')
qtest_s = qtest.split('\n')
qdev_s = qdev.split('\n')
queries_train = pd.DataFrame(qtrain_s)
queries_test = pd.DataFrame(qtest_s)
queries_dev = pd.DataFrame(qdev_s)

queries_train[['_id', 'text']] = queries_train[0].str.split('\t', expand=True)
queries_test[['_id', 'text']] = queries_test[0].str.split('\t', expand=True)
queries_dev[['_id', 'text']] = queries_dev[0].str.split('\t', expand=True)
queries_train = queries_train.drop(columns=[0])
queries_test = queries_test.drop(columns=[0])
queries_dev = queries_dev.drop(columns=[0])
queries_train = queries_train.drop(queries_train.index[-1])
queries_test = queries_test.drop(queries_test.index[-1])
queries_dev = queries_dev.drop(queries_dev.index[-1])

queries_train.to_json('queries_train.json', orient='records', lines=True)
queries_test.to_json('queries_test.json', orient='records', lines=True)
queries_dev.to_json('queries_dev.json', orient='records', lines=True)

#### DOCS ####
with open('/home/ubuntu/esokli/DESIRE-ME/nfcorpus/train.docs', 'r') as file:
    dtrain = file.read()
with open('/home/ubuntu/esokli/DESIRE-ME/nfcorpus/test.docs', 'r') as file:
    dtest = file.read()
with open('/home/ubuntu/esokli/DESIRE-ME/nfcorpus/dev.docs', 'r') as file:
    ddev = file.read()

dtrain_s = dtrain.split('\n')
dtest_s = dtest.split('\n')
ddev_s = ddev.split('\n')
docs_train = pd.DataFrame(dtrain_s)
docs_test = pd.DataFrame(dtest_s)
docs_dev = pd.DataFrame(ddev_s)

docs_train[['_id', 'text']] = docs_train[0].str.split('\t', expand=True)
docs_test[['_id', 'text']] = docs_test[0].str.split('\t', expand=True)
docs_dev[['_id', 'text']] = docs_dev[0].str.split('\t', expand=True)
docs_train = docs_train.drop(columns=[0])
docs_test = docs_test.drop(columns=[0])
docs_dev = docs_dev.drop(columns=[0])
docs_train = docs_train.drop(docs_train.index[-1])
docs_test = docs_test.drop(docs_test.index[-1])
docs_dev = docs_dev.drop(docs_dev.index[-1])

docs_train.to_json('docs_train.json', orient='records', lines=True)
docs_test.to_json('docs_test.json', orient='records', lines=True)
docs_dev.to_json('docs_dev.json', orient='records', lines=True)