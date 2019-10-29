curl http://localhost:8983/solr/lee8/config/params -H 'Content-type:application/json' -d '{ "set":{ "test_params":{  "f": ["id:/id", "paragraph:/paragraph"]}}}'
