import pandas as pd
df2=pd.read_csv("PATH_TO_RESULT_CSV", sep="\t")
!pip install bert_score
import numpy as np
from tqdm import tqdm
from bert_score import score

np.random.seed(42)
BERT_Scores_general = []

# Assuming 'lang' is the language of your text, you need to replace it with the appropriate language code.
lang = 'en'  # Replace with the language code of your text

with tqdm(total=len(df2), position=0, leave=True) as pbar:
    for i in range(len(df2)):

        original_sent = df2['Original Debiased Sentence'].iloc[i] 

        prediction = df2['Predicted Debiased Sentence'].iloc[i] 

        # Check if prediction is not NaN or None
        if isinstance(prediction, str):
            # Compute BERTScore
            _, _, bert_score = score([original_sent], [prediction], lang=lang)
            # Extract F1 score (you can use other scores like precision or recall as needed)
            f1_score = bert_score.mean().item()
        else:
            # If prediction is missing or NaN, assign a default score
            f1_score = 0.0

        # Append score
        BERT_Scores_general.append(f1_score)
        pbar.set_postfix({'BERT Score': f1_score}, refresh=True)
        pbar.update(1)  # Update progress bar

print('Average BERTScore using general scoring is ===> ', np.mean(BERT_Scores_general))
