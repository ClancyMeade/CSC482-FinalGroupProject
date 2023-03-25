# CSC482-FinalGroupProject
- In this project we created a sentiment classifier to analyze current sentiment toward different tech companies based on recent tweets about them. 
- To run this project: 
    - Download required packages 
    - Download the embeddings 
    - Then do any of testing, training, or visualization with the dashboard 
    - NOTE: Training is not required, the pre-trained models are stored in the models directory and will be loaded automatically 

## Packages: 
- nltk
- numpy 
- keras 
- sklearn 

## Downloading Embeddings: 
- `python3 embeddings.py`

## Training All Models: 
- To run with k-fold: 
`python3 sentiment_analysis.py -train -k`
- To run without k-fold: 
`python3 sentiment_analysis.py -train`

## Testing All Models: 
- `python3 sentiment_analysis.py -test`

## Running The Dashboard
- `python3 app.py n`
- Where n is the number of tweets to fetch
- If n is not given, the default is 10
