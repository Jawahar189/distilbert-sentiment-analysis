cd C:\Users\HP\Desktop\sentiment_analysis
.\.venv\Scripts\activate  
python src/train.py        

python src/train.py --max-train-samples 2000 --max-eval-samples 1000 --epochs 1
python src/predict.py --text "This movie was absolutely fantastic."