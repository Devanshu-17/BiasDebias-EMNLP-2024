import nltk
nltk.download('wordnet')
import numpy as np
from tqdm import tqdm
from nltk.translate.meteor_score import meteor_score, single_meteor_score
import pandas as pd

np.random.seed(42)
METEOR_Scores_general = []

df2 = pd.read_csv("PATH_TO_RESULT_CSV", sep="\t")

for i in tqdm(range(len(df2))):

    original_sent = df2['Predicted Debiased Sentence'].iloc[i].split()

    prediction = df2['Original Debiased Sentence'].iloc[i].split() 

    # Compute METEOR score
    meteor_score_value = meteor_score([original_sent], prediction)

    # Append score
    METEOR_Scores_general.append(meteor_score_value)

print('Average METEOR score using general scoring is ===> ', np.mean(METEOR_Scores_general))