bin/solr start -p 8983
bin/solr create -c core_queries
bin/solr create -c core_paras
bin/post -c core_queries ./delete_all.xml
bin/post -c core_paras ./delete_all.xml
bin/post -c core_paras ../squad_data/xmled_squad/squad_paras.xml
bin/post -c core_queries ../squad_data/xmled_squad/squad_queries.xml
#curl "http://localhost:8983/solr/core_paras/select?fl=*%2Cscore&fq=is_query%3AF&q=content%3AWhat%27s%20the%20funny%20name%20for%20resin-saturated%20heartwood%3F&sow=false"

