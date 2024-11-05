# medapis-final-version


# MedApps-Docker-Phani-RAG
##### clone the repo and run the ollama_rag.ipynb file to get the db folder for RAG and then run the following command to start the docker container

```bash
docker compose up --build
```
##### for detached mode
```bash
docker compose up --build -d
```


```bash
docker pull 1681149pk/medapp:srmap
docker run -p 8000:8000 -p 8501:8501 -p 8502:8502 1681149pk/medapp:srmap
```
