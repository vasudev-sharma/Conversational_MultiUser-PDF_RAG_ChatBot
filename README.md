pip install -r requirements/requirements.in
pip install -r requirements/requirements.txt


## Local installation

**Ensure you have your OPENAI_API_KEY in `.env` file**
1. `python vector_loader.py`
2. `strealit run streamlit_apply.py`


# Deployment
## Deployment script (Minikube)
Before running this script, ensure that you have both `minikube` and `Docker` on your system.
1. `minikube start`
2. `docker build -t test_image .` : Build image locally
3. `minikube image load test_image`
4. `kubectl apply -f deployment.yaml`
5. `kubectl apply -f service.yaml`

![alt text](images/image.png) You can test the running status of pods and the container using `k9s`



## Testing
1. Expose the service port: kubectl expose deployment my-app --type=NodePort --port=8501`
2. `minikube service my-app --url` and click on the url / it will generate url like: `http://127.0.0.1:49154`
3. `curl <url>` / click on url on local machine

![alt text](images/streamlit.png)


# Monitoring Setup

## Grafana and Prometheus: Install using helm
- `helm repo add prometheus-community https://prometheus-community.github.io/helm-charts`
- `helm repo update`
- prom-operator (admin password)
- `helm install monitoring prometheus-community/kube-prometheus-stack -f values.yaml`
```
NAME: monitoring
LAST DEPLOYED: Mon May 26 17:34:47 2025
NAMESPACE: default
STATUS: deployed
REVISION: 1
NOTES:
kube-prometheus-stack has been installed. Check its status by running:
  kubectl --namespace default get pods -l "release=monitoring"

Get Grafana 'admin' user password by running:

  kubectl --namespace default get secrets monitoring-grafana -o jsonpath="{.data.admin-password}" | base64 -d ; echo

Access Grafana local instance:

  export POD_NAME=$(kubectl --namespace default get pod -l "app.kubernetes.io/name=grafana,app.kubernetes.io/instance=monitoring" -oname)
  kubectl --namespace default port-forward $POD_NAME 3000

Visit https://github.com/prometheus-operator/kube-prometheus for instructions on how to create & configure Alertmanager and Prometheus instances using the Operator.
```
![alt text](image.png)


## LLM Monitoring and Tracing
1. Langchain Tracing (Langsmith): Ensure you have added your langsmith API key and tracing setup in `.env` file.
![alt text](images/Langchain.png)




TODO: 
- Add multiple users
- Add alerts if no documents are retreived by the LLM. In production, integrate it with PagerDuty or Slack.

# Performance Optimization

1. Calling the FAISS DB once and not every time we are querying  (remove redundancy)
2. Building the FAISS index ourself and not using defaults
3. Error handling with FAISS DB
## Questions to answer:

### Vector DB
- issue with basic vector data: 
  - FAISS: fine-tune with k results and add a threshold
  - With small vectors, we can use IndexFlatL2 index. However, as the size of document grows, vectors scale up as well. If vector size reaches in the order of millions, then we should should other indexing approach such as Inverted File Index (IndexIVFFlat / IndexIVFPQ)
  - in-memory store, so slow as dataset size grows
  - not caching of frequent queries
  - no support for hybrid search
  -
- shift to Distributed vector database: Weviate DB and deploy to cloud
- 
## Improved Retreival and Ranking
- Memory of chatbot
- Better embedding 
  - properietary
  - open source:
      - BAAI/bge-base-en-v1.5 
      - sentence-transformers/all-MiniLM-L6-v2

- Try different chunking strategies
- Hybrid retreival (Keyword + Dense): https://python.langchain.com/docs/how_to/multi_vector/
  - BM25 + FAISS 


- Promt design 
  - Explicit instructions
  - Prevent hallucinations on the responses

## Scalability of the chatbot with large number of users
