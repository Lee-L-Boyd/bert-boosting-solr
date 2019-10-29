import pickle
import urllib
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2
import re
import requests
total = 0.
points = 0.

#connection = urllib2.urlopen('http://easel3.fulgentcorp.com:8983/solr/core_test1/select?fl=paragraph_id,content&fq=is_query:T&rows=10000000&q=*:*&fq=is_testing:T')
connection = urllib2.urlopen('http://localhost:8983/solr/core_queries/select?fl=id,paragraph_id,text&fq=is_query:T&rows=1000000&q=*:*&fq=is_testing:T&fq=is_impossible:F')
response = eval(connection.read())
query_pre = 'http://localhost:8983/solr/core_paras/select?'
bad_results = []
for r in response['response']['docs']:
  #content = 'content:' + re.sub('\s', ' or ', r['content'][0])
  #content = 'content:' + r['content'][0]
  #print(r.keys())
  content = r['text'][0]
  
  query_post = {'q' : content, 'fl' : 'para_id,score', 'fq': 'is_query:F', 'sow':'false', 'rows':'10'}
  query = urllib.parse.urlencode(query_post)
  query = query.encode('utf-8').decode('utf-8')
  query = query_pre+query
  #print(query)
  try:
    connection2 = urllib2.urlopen(query)
    paragraphs = eval(connection2.read())
    sanity_check = 0
    for p in paragraphs['response']['docs']:
      if r['paragraph_id'][0] == p['para_id'][0]:
        points += 1
        sanity_check+=1
        break
    if sanity_check > 1:
      print("Sanity check failed")
    if sanity_check == 0:
      bad_results.append((r['text'], r['paragraph_id']))
    sanity_check = 0
    total+=1
  except Exception as e:
    print(str(e))
    print(str(query))
print(points/total)
pickle.dump(bad_results, open('bad_results.pickle','wb'))
