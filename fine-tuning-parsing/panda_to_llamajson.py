import pandas as pd
import json
import numpy

def create_instruction_entry(instruction, input, scores):
    return {
            "instruction" : instruction,
            "input" : input,
            "output": f"\"Openness: {round(scores[0] / 7 * 5, 1)}, Extraversion: {round(scores[1] / 7 * 5, 1)},  Agreeableness: {round(scores[2]/ 7 * 5, 1)}, Conscientiousness: {round(scores[3] / 7 * 5, 1)}, Neuroticism: {round (scores[4] / 7 * 5, 1)}\""
            }
        #return "{\n" + f"\"instruction\" : \"{instruction}\",\n \"input\": \"{input}\",\n \"output\": \"Openness: {scores[0]}, Extraversion: {scores[1]},  Agreeableness: {scores[2]}, Conscientiousness: {scores[3]}, Neuroticism: {scores[4]}\"" + "\n}"

instruction_prompt = "You are a psychologist experienced in analysizing text and determining their personality profile. Your task will be to classify user text based on the OCEAN model, or Big Five personality traits model. From the text, you want to determine five traits:\nConscientiousness: order, dutifulness, achievement striving, self-discipline, deliberation.\nAgreeablesness: trust, straightforwardness, altruism, compliance, modesty, tendermindedness.\nNeuroticism: anxiety, angry hostility, depression, self-consciousness, impulsiveness, vulnerability.\nOpenness: fantasy, aesthetics, values, positive attitude towards new experiences.\nExtraversion: warmth, gregariousness, assertiveness, excitement-seeking.\nInstructions: Read the individual\'s text very carefully. For each personality trait, assign a score between 1 (Very Low) and 5 (Very High) based on the themes, tone, and content of the responses. Each score can be rounded to the nearest tenth, like 3.5.\nScoring Guide:\n1: Very Low - The response shows little to no alignment with the trait's facets.\n2: Low - The response shows weak alignment with the trait's facets.\n3: Moderate - The response shows some alignment but not strongly.\n4: High - The response strongly aligns with the trait's facets.\n5: Very High - The response shows exceptional alignment with the trait's facets.\nAnalyze the following text based on the Big Five personality traits, and output the scores without any extra texts or explanation using the following format\n \"Openness: score, Extraversion: score, Agreeablesness: score, Conscientiousness: score, Neuroticism: score\":"

file = open("./out.json", "a+")
df = pd.read_csv("hf://datasets/MTHR/OCEAN/OCEAN-synthetic.csv")
df.reset_index()

#res = "[\n"
instruction_list = list()

for _, row in df.iterrows():
        if row.isna().any():
            continue
        instruction_list.append(create_instruction_entry(instruction_prompt, row.iloc[0], row.iloc[1:6]))



json.dump(instruction_list, file, indent="")
#res += '\n'.join(numpy.array(instruction_list)) + "\n]"
#file.write(res)
#json.load(file)
file.close()




