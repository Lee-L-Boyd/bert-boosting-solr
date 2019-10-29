from bert_serving.client import BertClient
import torch
bc = BertClient()
bc_para = BertClient(port=5557, port_out=5558)
bertized_query = torch.tensor(bc.encode(['This is a test'], show_tokens=True)[0])
#print(bertized_query.view(1,50,9,-1).size())
print(bertized_query.size())

bertized_query_para = torch.tensor(bc_para.encode(['This is a test']))
print(bertized_query_para.size())


