import pandas as pd
import json
import numpy
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

def create_instruction_entry(instruction, input, scores):
    return {
            "instruction" : instruction,
            "input" : input,
            #"output": f"\"OPN: {scores[4]}, EXT: {scores[0]}, AGR: {scores[2]}, CON: {scores[3]}, NEU: {scores[1]}\""
            "output": f"{round(scores[0] / 7 * 5, 1)}, {round(scores[1] / 7 * 5, 1)}, {round(scores[2]/ 7 * 5, 1)}, {round(scores[3] / 7 * 5, 1)}, {round (scores[4] / 7 * 5, 1)}"
            }
        #return "{\n" + f"\"instruction\" : \"{instruction}\",\n \"input\": \"{input}\",\n \"output\": \"Openness: {scores[0]}, Extraversion: {scores[1]},  Agreeableness: {scores[2]}, Conscientiousness: {scores[3]}, Neuroticism: {scores[4]}\"" + "\n}"

PROMPT = "## INSTRUCTIONS \nYou are a psychologist experienced in analyzing text and determining a person's personality profile based on its contents. Your task will be to classify user text based on the OCEAN model, or Big Five personality traits model. \n\n## SCORING GUIDE \n1: Very Low - The response shows little to no alignment with the trait's facets.\n2: Low - The response shows weak alignment with the trait's facets.\n3: Moderate - The response shows some alignment but not strongly.\n4: High - The response strongly aligns with the trait's facets.\n5: Very High - The response shows exceptional alignment with the trait's facets.\n\n## TASK\nRead the user's text carefully, and for each personality trait, assign a score between 1 (Very Low) and 5 (Very High) based on the content, tone and generally sentiment of the text. Each score can be rounded to the nearest tenth. OUTPUT Output the score of for the personality traits in the following format: \"OPN, EXT, AGR, CON, NEU\". Do not specify which personality label the score is associated with. Do not provide any additional text or explanation in the result. \n\n## TEXT\n"
#instruction_prompt = "You are a psychologist experienced in analysizing text and determining their personality profile. Your task will be to classify user text based on the OCEAN model, or Big Five personality traits model. From the text, you want to determine five traits: \nOpenness: How open a person is to new experiences, and to allowing their imagination to run wild. \nExtraversion: How energetic and socially outgoing a person, tending to be more talkative and assertive in conversation.\nAgreeableness: A person who exhibits prosocial behavior, including trust, kindness and affection.\nConscientiousness: A person who has high level of thoughtfulness, self-control, focused on a goal and organized.\nNeuroticism: A person who shoes moody behaviour, sadness, and unstable emotions, generally is negative.\nYou will classify a user's text based on these five personalities traits. for each trait, assign a score on a scale from 1 to 7, with a score of 1 meaning that the text does not align with the trait's values, and 7 meaning that the text aligns extremely well with the trait's values.\n Analyze the following text based on the Big Five personality traits, and output the scores without any extra texts or explanation using the following format \"Openness: score, Extraversion: score,  Agreeableness: score, Conscientiousness: score, Neuroticism: score\":"

splits = {'train': 'wcpr13-train.csv', 'test': 'wcpr13-test.csv'}
with open("./out_train.json", "a+") as file:
    df = pd.read_csv("hf://datasets/facells/facebook-personality-recognition-wcpr13/" + splits["train"])
    df.reset_index()

#res = "[\n"
    instruction_list = list()
    
    df = df.groupby(['AUTHID', 'scoreEXT', 'scoreNEU', 'scoreAGR', 'scoreCON', 'scoreOPN'])['STATUS'].apply(lambda x: ' '.join(x)).reset_index()
    print(df)
    for _, row in df.iterrows():
        if row.isna().any():
            continue
        instruction_list.append(create_instruction_entry(PROMPT, ILLEGAL_CHARACTERS_RE.sub('',row.iloc[6].strip()), row.iloc[1:6]))
    json.dump(instruction_list, file, indent="")

with open("./out_eval.json", "a+") as file:
    df = pd.read_csv("hf://datasets/facells/facebook-personality-recognition-wcpr13/" + splits["test"])
    df.reset_index()

#res = "[\n"
    instruction_list = list()
    
    df = df.groupby(['AUTHID', 'scoreEXT', 'scoreNEU', 'scoreAGR', 'scoreCON', 'scoreOPN'])['STATUS'].apply(lambda x: ' '.join(x)).reset_index()
    print(df)
    for _, row in df.iterrows():
        if row.isna().any():
            continue
        instruction_list.append(create_instruction_entry(PROMPT, ILLEGAL_CHARACTERS_RE.sub('',row.iloc[6].strip()), row.iloc[1:6]))
    json.dump(instruction_list, file, indent="")

#res += '\n'.join(numpy.array(instruction_list)) + "\n]"
#file.write(res)
#json.load(file)

