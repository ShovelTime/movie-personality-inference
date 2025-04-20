import json
import array
import sys
import re
import pandas as pd
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openai import OpenAI

MAX_TOKEN_COUNT = 12880 * 0.70 # maximum token count that the model can read at once. Divided by 3 to allow space for reasoning
PROMPT_TOKEN_COUNT = 422 # token space taken by the system prompt
PROMPT = "You are a psychologist experienced in analysizing text and determining their personality profile. Your task will be to classify user text based on the OCEAN model, or Big Five personality traits model. From the text, you want to determine five traits: \nOpenness: How open a person is to new experiences, and to allowing their imagination to run wild. \nExtraversion: How energetic and socially outgoing a person, tending to be more talkative and assertive in conversation.\nAgreeableness: A person who exhibits prosocial behavior, including trust, kindness and affection.\nConscientiousness: A person who has high level of thoughtfulness, self-control, focused on a goal and organized.\nNeuroticism: A person who shoes moody behaviour, sadness, and unstable emotions, generally is negative.\nYou will classify a user's text based on these five personalities traits. for each trait, assign a score on a scale from 1 to 7, with a score of 1 meaning that the text does not align with the trait's values, and 7 meaning that the text aligns extremely well with the trait's values. Values can be rounded up to a tenth, like 4.5 for example. \n Analyze the following text based on the Big Five personality traits, and output the scores without any extra texts or explanation using the following format \"Openness: score, Extraversion: score,  Agreeableness: score, Conscientiousness: score, Neuroticism: score\":"

class Reviewer:
    def __init__(self):
        self.reviewerID = None
        self.movies_watched = dict()
        self.reviewTexts = ""


def parse_completion(completion, user_id):
    res = list()
    res.append(user_id)
    str_arr = completion.split(',')
    for r_string in str_arr:
        s_split = r_string.split(':')[1]
        res.append(s_split)

    return res

def decompose_movie_map(movie_map, user_map):
    res = list()
    for movie_id in movie_map:
        user_set = movie_map[movie_id]
        for user_id in user_set:
            user = user_map[user_id]
            res.append([movie_id, user_id, user.movies_watched[movie_id]])

    return res

def decompose_user_map(user_map): # TODO: add in personality matrix
    res = list()
    for user in user_map.values():
        res.append([user.reviewerID, ILLEGAL_CHARACTERS_RE.sub(r'',user.reviewTexts.strip())])

    return res

def k_core_filter_pass(user_map, movie_map, k):
    pending_removal = list()
    filtered = False
    for movie_id in movie_map.keys():
        movie = movie_map[movie_id]
        if len(movie) < k:
            pending_removal.append(movie_id)
            filtered = True
            for user_id in movie:
                user = user_map[user_id]
                try:
                    del user.movies_watched[movie_id]
                except KeyError:
                    print("User does not have movie?\n\n")
                    print(str(len(user.movies_watched)))
                if len(user.movies_watched) == 0:
                    del user_map[user_id]

    for movie_id in pending_removal:
        del movie_map[movie_id]
    pending_removal.clear()

    for user in user_map.values():
        if len(user.movies_watched) < k:
            filtered = True
            pending_removal.append(user.reviewerID)
    for user_id in pending_removal:
        user = user_map[user_id]
        for movie_id in user.movies_watched.keys():
            movie = movie_map[movie_id]
            movie.remove(user_id)
        del user_map[user_id]

    return filtered

def perform_personality_inference(user_map, movie_map):
        personality_map = list()
        #OpenAI
        openai_key = "EMPTY"
        openai_base = "http://localhost:8000/v1"
        openai_client = OpenAI(
            api_key=openai_key,
            base_url=openai_base,
        )

        print("Beginning personality inference...")

        for user in user_map.values():
            user_id = user.reviewerID
            input = user.reviewTexts
            print("Prompt: \n\n" + PROMPT + "\n" + input)
            res = openai_client.chat.completions.create(
                model="unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
                messages=[
                   {"role": "system", "content": PROMPT},
                    {"role": "user", "content": input},
                ]
            )
            print("\nOutput:\n\n" + res.choices[0].message.content)
            personality_m = parse_completion(res.choices[0].message.content, user_id)
            personality_map.append(personality_m)

        print("Writing excel to /tmp/dataset")

        movie_frame = decompose_movie_map(movie_map, user_map)
        user_frame = decompose_user_map(user_map) #, personality_m) work personality matrix in eventually

        user_dataframe = pd.DataFrame(personality_map, columns=["user_id", "openness", "extraversion", "agreeableness", "conscientiousness", "neuroticism"])
        movie_dataframe = pd.DataFrame(movie_frame, columns=["movie_id", "user_id", "score"])

        with pd.ExcelWriter("/tmp/dataset/out.xlsx") as excel_out:
            user_dataframe.to_excel(excel_out, sheet_name="User")
            movie_dataframe.to_excel(excel_out, sheet_name="Movie")



def main():
    user_map = dict()
    movie_map = dict()
    with open("/tmp/dataset/new.json", "r") as file:
        json_file = json.load(file)
        for json_item in json_file:
            if "reviewText" not in json_item:
                continue
            user_id = json_item["reviewerID"]
            movie_id = json_item["asin"]
            user = None
            if user_id in user_map:
                user = user_map[user_id]
            else:
                user = Reviewer()
                user.reviewerID = user_id
            user.movies_watched[movie_id] = json_item["overall"]
            user.reviewTexts += json_item["reviewText"] + "\n\n"
            
            user_set = None
            if movie_id in movie_map:
                user_set = movie_map[movie_id]
            else:
                user_set = set()

            user_set.add(user_id)
            movie_map[movie_id] = user_set
            user_map[user_id] = user

        print("Finished parsing reviews, user count = " + str(len(user_map)) + "\nmovie count:" + str(len(movie_map)))
        
        #20-core filtering
        while k_core_filter_pass(user_map, movie_map, 20) != False:
            continue


        print("\n\nProcessed 20-core filter passes, user count = " + str(len(user_map)) + "\nmovie count:" + str(len(movie_map)))


    average = 0
    highest = 0
    lowest = 0
    for user in user_map.values():
        length = len(re.findall(r'\w+', user.reviewTexts))
        average += length
        if length < lowest or lowest == 0:
            lowest = length
        if length > highest or highest == 0:
            highest = length

    average = average / len(user_map.keys())

    print(f"Review Text Word Count Statistics:\naverage : {average}\nhighest : {highest}\nlowest : {lowest}\n")


    #perform_personality_inference(user_map, movie_map)


    input("Press Enter to Continue....")
    return 0

if __name__ == '__main__':
    sys.exit(main())

