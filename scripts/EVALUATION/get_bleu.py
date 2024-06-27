import numpy as np
from tqdm import tqdm
from nltk.translate import bleu_score
import pandas as pd

df2 = pd.read_csv("PATH_TO_RESULT_CSV", sep="\t")


BLEU_Scores_general = []

for i in tqdm(range(len(df2))):

    original_sent = df2['Predicted Debiased Sentence'].iloc[i].split() 

    prediction = df2['Original Debiased Sentence'].iloc[i].split() 

    # Compute BLEU score
    if len(original_sent) == 0:
        blue_score = 0.0
    else:
        blue_score = bleu_score.sentence_bleu([original_sent], prediction)

    # Append score
    BLEU_Scores_general.append(blue_score)

print('Average BLEU score using general scoring is ===> ', np.mean(BLEU_Scores_general))