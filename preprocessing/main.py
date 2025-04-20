import json
import sys
import pandas as pd
from openai import OpenAI

class PersonalityMatrix:
    def __init__(self):
        self.open = 0.0
        self.extra = 0.0
        self.agree = 0.0
        self.consc = 0.0
        self.neuro = 0.0

    def input_values(vals):
        self.open = vals[0]
        self.extra = vals[1]
        self.agree = vals[2]
        self.consc = vals[3]
        self.neuro = vals[4]


class Review:
    def __init__(self, movieID, userID, score):
        self.movieID = movieID
        self.userID = userID
        self.score = score

class Reviewer:
    def __init__(self):
        self.reviewerID = None
        self.movies_watched = dict()
        self.reviewTexts = ""


def parse_completion(completion):
    res_arr = array('f')
    str_arr = completion.split(',')
    for r_string in str_arr:
        s_split = r_string.split(':')[1]
        res_arr.append(s_split)

    res = PersonalityMatrix()
    res.input_values(res_arr)
    return res


def decompose_movie_map(movie_map, user_map):
    res = list()

    for movie_id in movie_map:
        user_set = movie_map[movie_id]
        for user_id in user_set:
            user = user_map[user_id]
            res.append(Review(movie_id, user_id, user.movies_watched[movie_id]))
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
        
        #10-core filtering
        while k_core_filter_pass(user_map, movie_map, 25) != False:
            continue


        print("\n\nProcessed 25-core filter passes, user count = " + str(len(user_map)) + "\nmovie count:" + str(len(movie_map)))

#    personality_map = dict()

    #OpenAI
#    openai_key = "EMPTY"
#    openai_base = "http://localhost:8000/v1"

#    openai_client = OpenAI(
#        api_key=openai_key,
#        base_url=openai_base,
#    )

#    prompt = "You are a psychologist experienced in analysizing text and determining their personality profile. Your task will be to classify user text based on the OCEAN model, or Big Five personality traits model. From the text, you want to determine five traits: \nOpenness: How open a person is to new experiences, and to allowing their imagination to run wild. \nExtraversion: How energetic and socially outgoing a person, tending to be more talkative and assertive in conversation.\nAgreeableness: A person who exhibits prosocial behavior, including trust, kindness and affection.\nConscientiousness: A person who has high level of thoughtfulness, self-control, focused on a goal and organized.\nNeuroticism: A person who shoes moody behaviour, sadness, and unstable emotions, generally is negative.\nYou will classify a user's text based on these five personalities traits. for each trait, assign a score on a scale from 1 to 7, with a score of 1 meaning that the text does not align with the trait's values, and 7 meaning that the text aligns extremely well with the trait's values. Values can be rounded up to a tenth, like 4.5 for example. \n Analyze the following text based on the Big Five personality traits, and output the scores without any extra texts or explanation using the following format \"Openness: score, Extraversion: score,  Agreeableness: score, Conscientiousness: score, Neuroticism: score\":"

#    print("Beginning personality inference...")

#    for user in user_map.values():
#        userID = user.reviewerID
#        input = user.reviewTexts
#        res = openai_client.completions.create(
#            model="unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
#            messages=[
#               {"role": "system", "content": prompt},
#                {"role": "user", "content": input},
#            ]
#        )
#        personality_m = parse_completion(res)
#        personality_map[userID] = personality_m

    print("Writing excel to /tmp/dataset")

    movie_frame = decompose_movie_map(movie_map, user_map)

    user_dataframe = pd.DataFrame(user_map, index=[0,3])
    movie_dataframe = pd.DataFrame(movie_frame)

    with pd.ExcelWriter("/tmp/dataset/out.xlsx") as excel_out:
        user_dataframe.to_excel(excel_out, sheet_name="User")
        movie_dataframe.to_excel(excel_out, sheet_name="Movie")



    input("Press Enter to Continue....")
    return 0

if __name__ == '__main__':
    sys.exit(main())

