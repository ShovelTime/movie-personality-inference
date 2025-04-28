import pandas as pd
import json
import numpy

def create_instruction_entry(instruction, input, scores):
    return {
            "instruction" : instruction,
            "input" : input,
            "output": f"\"Openness: {round(scores[1] / 100.0 * 5.0, 1)}, Extraversion: {round(scores[3] / 100.0 * 5.0, 1)},  Agreeableness: {round(scores[0] / 100.0 * 5.0, 1)}, Conscientiousness: {round(scores[2] / 100.0 * 5.0,2)}, Neuroticism: {round(scores[4] / 100.0 * 5.0, 1)}\""
            }
        #return "{\n" + f"\"instruction\" : \"{instruction}\",\n \"input\": \"{input}\",\n \"output\": \"Openness: {scores[0]}, Extraversion: {scores[1]},  Agreeableness: {scores[2]}, Conscientiousness: {scores[3]}, Neuroticism: {scores[4]}\"" + "\n}"

PROMPT = "You are a psychologist experienced in analysizing text and determining their personality profile. Your task will be to classify user text based on the OCEAN model, or Big Five personality traits model. From the text, you want to determine five traits:\nConscientiousness: order, dutifulness, achievement striving, self-discipline, deliberation.\nAgreeablesness: trust, straightforwardness, altruism, compliance, modesty, tendermindedness.\nNeuroticism: anxiety, angry hostility, depression, self-consciousness, impulsiveness, vulnerability.\nOpenness: fantasy, aesthetics, values, positive attitude towards new experiences.\nExtraversion: warmth, gregariousness, assertiveness, excitement-seeking.\nInstructions: Read the individual\'s text very carefully. For each personality trait, assign a score between 1 (Very Low) and 5 (Very High) based on the themes, tone, and content of the responses. Each score can be rounded to the nearest tenth, like 3.5.\nScoring Guide:\n1: Very Low - The response shows little to no alignment with the trait's facets.\n2: Low - The response shows weak alignment with the trait's facets.\n3: Moderate - The response shows some alignment but not strongly.\n4: High - The response strongly aligns with the trait's facets.\n5: Very High - The response shows exceptional alignment with the trait's facets.\nAnalyze the following text based on the Big Five personality traits, and output the scores without any extra texts or explanation using the following format\n \"Openness: score, Extraversion: score, Agreeablesness: score, Conscientiousness: score, Neuroticism: score\":"

splits = {'train': 'Personality Datasets - Reddit/train_set.csv', 'validation': 'Personality Datasets - Reddit/val_set.csv', 'test': 'Personality Datasets - Reddit/eval_set.csv'}

with open("./out_train.json", "a+") as file:
    df = pd.read_csv("hf://datasets/Fatima0923/Automated-Personality-Prediction/" + splits["train"])
    df.reset_index()

#res = "[\n"
    instruction_list = list()

    for _, row in df.iterrows():
        if row.isna().any():
            continue
        instruction_list.append(create_instruction_entry(PROMPT, row.iloc[0], row.iloc[1:6]))

    json.dump(instruction_list, file, indent="")


with open("./out_val.json", "a+") as file:
    df = pd.read_csv("hf://datasets/Fatima0923/Automated-Personality-Prediction/" + splits["validation"])
    df.reset_index()

#res = "[\n"
    instruction_list = list()

    for _, row in df.iterrows():
        if row.isna().any():
            continue
        instruction_list.append(create_instruction_entry(PROMPT, row.iloc[0], row.iloc[1:6]))

    json.dump(instruction_list, file, indent="")

#res += '\n'.join(numpy.array(instruction_list)) + "\n]"
#file.write(res)
#json.load(file)

