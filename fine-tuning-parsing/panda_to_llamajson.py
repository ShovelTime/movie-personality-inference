import pandas as pd
import json
import numpy

def create_instruction_entry(instruction, input, scores):
    return {
            "instruction" : instruction,
            "input" : input,
            #"output": f"\"OPN: {round(scores[0] / 7 * 5, 1)}, EXT: {round(scores[1] / 7 * 5, 1)},  AGR: {round(scores[2]/ 7 * 5, 1)}, CON: {round(scores[3] / 7 * 5, 1)}, NEU: {round (scores[4] / 7 * 5, 1)}\""
            "output": f"{round(scores[0] / 7 * 5, 1)}, {round(scores[1] / 7 * 5, 1)}, {round(scores[2]/ 7 * 5, 1)}, {round(scores[3] / 7 * 5, 1)}, {round (scores[4] / 7 * 5, 1)}"
            }
        #return "{\n" + f"\"instruction\" : \"{instruction}\",\n \"input\": \"{input}\",\n \"output\": \"Openness: {scores[0]}, Extraversion: {scores[1]},  Agreeableness: {scores[2]}, Conscientiousness: {scores[3]}, Neuroticism: {scores[4]}\"" + "\n}"

instruction_prompt = "## INSTRUCTIONS \nYou are a psychologist experienced in analyzing text and determining a person's personality profile based on its contents. Your task will be to classify user text based on the OCEAN model, or Big Five personality traits model. \n\n## SCORING GUIDE \n1: Very Low - The response shows little to no alignment with the trait's facets.\n2: Low - The response shows weak alignment with the trait's facets.\n3: Moderate - The response shows some alignment but not strongly.\n4: High - The response strongly aligns with the trait's facets.\n5: Very High - The response shows exceptional alignment with the trait's facets.\n\n## TASK\nRead the user's text carefully, and for each personality trait, assign a score between 1 (Very Low) and 5 (Very High) based on the content, tone and generally sentiment of the text. Each score can be rounded to the nearest tenth. OUTPUT Output the score of for the personality traits in the following format: \"Openness, Extroversion, Agreeableness, Conscientiousness, Neuroticism\". Do not specify which personality label the score is associated with. Do not provide any additional text or explanation in the result. \n\n## TEXT\n"

file = open("./ocean_synthetic.json", "a+")
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




