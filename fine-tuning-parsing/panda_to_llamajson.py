import pandas as pd
import json
import numpy

def create_instruction_entry(instruction, input, scores):
    return {
            "instruction" : instruction,
            "input" : input,
            "output": f"\"Openness: {scores[0]}, Extraversion: {scores[1]},  Agreeableness: {scores[2]}, Conscientiousness: {scores[3]}, Neuroticism: {scores[4]}\""
            }
        #return "{\n" + f"\"instruction\" : \"{instruction}\",\n \"input\": \"{input}\",\n \"output\": \"Openness: {scores[0]}, Extraversion: {scores[1]},  Agreeableness: {scores[2]}, Conscientiousness: {scores[3]}, Neuroticism: {scores[4]}\"" + "\n}"

instruction_prompt = "You are a psychologist experienced in analysizing text and determining their personality profile. Your task will be to classify user text based on the OCEAN model, or Big Five personality traits model. From the text, you want to determine five traits: \nOpenness: How open a person is to new experiences, and to allowing their imagination to run wild. \nExtraversion: How energetic and socially outgoing a person, tending to be more talkative and assertive in conversation.\nAgreeableness: A person who exhibits prosocial behavior, including trust, kindness and affection.\nConscientiousness: A person who has high level of thoughtfulness, self-control, focused on a goal and organized.\nNeuroticism: A person who shoes moody behaviour, sadness, and unstable emotions, generally is negative.\nYou will classify a user's text based on these five personalities traits. for each trait, assign a score on a scale from 1 to 7, with a score of 1 meaning that the text does not align with the trait's values, and 7 meaning that the text aligns extremely well with the trait's values.\n Analyze the following text based on the Big Five personality traits, and output the scores without any extra texts or explanation using the following format \"Openness: score, Extraversion: score,  Agreeableness: score, Conscientiousness: score, Neuroticism: score\":"

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




