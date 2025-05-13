import json
import array
import sys
import re
import warnings
import pandas as pd
import os.path
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
#from openai import OpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer, RobertaTokenizerFast
import torch
from torch.utils.data import Dataset, DataLoader
from multiprocessing.pool import Pool, ThreadPool
from sklearn import preprocessing
from collections import defaultdict
import time
import numpy as np
import random
from tqdm import tqdm


#Control Random Indices selection for split,
SPLIT_RANDOM_SEED = 32489634

class Reviewer:
    def __init__(self):
        self.reviewerID = None
        self.movies_watched = dict()
        self.reviewTexts = dict()

    def to_dict(self):
        return {
            "userID": self.reviewerID,
            "moviesWatched": self.movies_watched,
            "reviewTexts": self.reviewTexts,
        }
    
def user_from_dict(user_id, movie_set, review_text):
    res = Reviewer()
    res.reviewerID = user_id
    res.movies_watched = movie_set
    res.reviewTexts = review_text
    return res

trait_names = ["agreeableness", "openness", "conscientiousness", "extraversion", "neuroticism"]

def parse_completion(completion):
    res = [0.0,0.0,0.0,0.0,0.0]
    for trait, score in zip(trait_names, completion):
        match trait.strip():
            case "openness":
                res[0] = score
                continue
            case "extraversion":
                res[1] = score
                continue
            case "agreeableness":
                res[2] = score
                continue
            case "conscientiousness":
                res[3] = score
                continue
            case "neuroticism":
                res[4] = score
                continue

    return res

def personality_average(p_list):
    res = [0.0, 0.0, 0.0, 0.0, 0.0]
    for p_mat in p_list:
        res[0] += p_mat[0]
        res[1] += p_mat[1]
        res[2] += p_mat[2]
        res[3] += p_mat[3]
        res[4] += p_mat[4]
    
    res[0] /= len(p_list)
    res[1] /= len(p_list)
    res[2] /= len(p_list)
    res[3] /= len(p_list)
    res[4] /= len(p_list)

def decompose_movie_map(movie_map, user_map, personality_map):
    res = list()
#    for movie_id in movie_map:
#        user_set = movie_map[movie_id]
#        for user_id in user_set:
#            user = user_map[user_id]
#            final = [movie_id, user_id, user.movies_watched[movie_id][0], user.movies_watched[movie_id][1]]
#            final.extend(personality_map[user_id])
#            res.append(final)
#
#    return res
    for user_id, user in user_map.items():
        for movie_id, rating in user.movies_watched.items():
            final = [movie_id, user_id, rating[0], rating[1]]
            final.extend(personality_map[user_id])
            res.append(final)
    return res

def decompose_user_map(user_map): # TODO: add in personality matrix
    eval_user_map = dict()
    res = list()
    for user in user_map.values():
        res.append([user.reviewerID, user.reviewTexts])

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
                    del user.reviewTexts[movie_id]
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

#def get_duplicates(snippet, text):
#    res = list()
#    if snippet == None:
#        return None
#    for n in range(0, len(text)):
#        if token_list[n] == snippet:
#            res.append(n)
#    return res


def split_review_text(text, tokenizer):
    res = list()
    current_text = text
    pre_tokens = tokenizer.tokenize(text)
    tokens = list()
    for token in pre_tokens:
        if token == "Ġ":
            res.append(' ')
        new = token.replace('Ġ', '')
        tokens.append(new)

    if len(tokens) <= 64:
        return None

    while len(tokens) > 64:
        #index = -1
        slice = tokens[0:64]
        last_token = slice[-1]

        #duplicates = get_duplicates(last_token, 64, slice)
        #if len(duplicates) == 0:
        #    print(last_token)
        #    print(slice)
        #    raise Exception("Get duplicates Failed")
        #if len(duplicates) > 1:
        #    search_string = current_text
        #    for duplicate in duplicates:
        #        index = search_string.find(last_token)
        #        if index == -1:
        #            print(last_token)
        #            print(slice)
        #            raise Exception("Find duplicates Failed")
        #        cutoff = index + len(last_token) + 1
        #        search_string = search_string[cutoff:]
        #else:
        #    index = text.find(last_token)
        #index = 

        if index == -1:
            print(last_token)
            print(slice)
            raise Exception("Find index Failed")
        cutoff = index + len(last_token) + 1
        split_text = current_text[:cutoff]
        res.append(split_text)
        current_text = current_text[cutoff:]
        tokens = tokens[64:]
    if len(current_text) != 0:
        res.append(current_text)
    return res

        


def perform_prediction(user_map):

    print("Performing Prediction")
    le = preprocessing.LabelEncoder()
    #users = list(user_map.values())
    users = np.array_split(list(user_map.values()), 10)
    total_count = len(users)
    device = torch.device('cuda')
    warnings.filterwarnings('ignore')
    
    model = AutoModelForSequenceClassification.from_pretrained("KevSun/Personality_LM", ignore_mismatched_sizes=True)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("KevSun/Personality_LM")
    

    personality_map = dict()
    
    chunk_nr = 0
    
    for chunks in users:
        
        chunk_nr += 1
        total_count = len(chunks)
        pbar = tqdm(total=total_count)
        print(f"Chunk {chunk_nr}/10")
        cur_count = 0
        encoded_users = list()
        encoded_input_ids = list()
        encoded_attn = list()
        for user in chunks:
            encoded = None
            for text in list(user.reviewTexts.values()):
                #text_split = split_review_text(text, tokenizer)
                #if text_split == None:
                #tokenized = tokenizer.tokenize(text)
            #    if len(tokenized) <= 64
            #        user_texts.append(text)
            #    else:
            #        split_tokenized = np.array_split(tokenized, 64)
            #        user_texts.extend(split_tokenized)
                tokenized = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=64, stride=2, return_overflowing_tokens=True)
                #if len(tokenizer.tokenize(text)) <= 64:
                #    print(encoded)
                del tokenized['overflow_to_sample_mapping']
                if encoded == None:
                    encoded = tokenized
                else:
                    encoded.input_ids = torch.cat((encoded.input_ids, tokenized.input_ids), 0)
                    encoded.attention_mask = torch.cat((encoded.attention_mask, tokenized.attention_mask), 0)
            if encoded.input_ids.size(0) > 100:
                chunked_input = torch.split(encoded.input_ids, 100, dim=0)
                chunked_attn = torch.split(encoded.attention_mask, 100, dim=0)
                encoded_users.append((user.reviewerID, True))
                encoded_input_ids.append(chunked_input)
                encoded_attn.append(chunked_attn)
            else:
                encoded_users.append((user.reviewerID, False))
                encoded_input_ids.append(encoded.input_ids)
                encoded_attn.append(encoded.attention_mask)
            pbar.update(1)
            #print(f"Encoding : {cur_count}/{total_count}")
       
        
        #encoded_input_ids = [t.to(device) for t in encoded_input_ids]
        #encoded_attn = [t.to(device) for t in encoded_attn]
        pbar.close()
        pbar = tqdm(total=total_count)
        cur_count = 0
        for user_info, input_ids, attn in zip(encoded_users, encoded_input_ids, encoded_attn):

            user_id, is_chunked = user_info
            if is_chunked == True:
                predictions_logits = None
                for chunked_input, chunked_attn in zip (input_ids, attn):
                    batch = {
                    "input_ids":      chunked_input.to(device),
                    "attention_mask": chunked_attn.to(device),
                    }
                    with torch.no_grad():
                        outputs = model(**batch) #Query model
                    if predictions_logits == None:
                        predictions_logits = outputs.logits.to('cpu')
                    else:
                        predictions = outputs.logits.to('cpu')
                        predictions_logits = torch.cat((predictions_logits, predictions), 0)

                predictions = predictions_logits.mean(dim=0) 
                predicted_scores = torch.nn.functional.softmax(predictions_logits, dim=-1).tolist()
                print(predicted_scores)
                #predicted_scores = predictions.mean(dim=0).tolist()
                personality = parse_completion(predicted_scores)
                personality_map[user_id] = personality
            else:
                batch = {
                    "input_ids":      input_ids.to(device),
                    "attention_mask": attn.to(device),
                }
                with torch.no_grad():
                    outputs = model(**batch) #Query model
                
                predictions = outputs.logits.to('cpu').mean(dim=0)
                predicted_scores = torch.nn.functional.softmax(predictions, dim=-1)
                #predicted_scores = predictions.mean(dim=0).tolist()
                print(predicted_scores)
                personality = parse_completion(predicted_scores)
                personality_map[user_id] = personality
                #print(f"User : {cur_count}/{total_count}")
            pbar.update(1)
        pbar.close()
    return personality_map


    #user_labels = torch.tensor(le.fit_transform(users))
    #encoded_map = list()
    #for ind in user_labels:
    #    user = user_map[users[ind.item()]]
    #    encodings = tokenizer(user.reviewTexts, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    #    #print(encodings)
    #    #raise Exception("HEHE CRASH")
    #    encoded_map.append((ind, encodings))
    #    cur_count += 1
    #    user.reviewTexts.clear()
    #    print(f"Encoding : {cur_count}/{total_count}")

    #raise Exception("HEHE")

    #dataset    = UserReviewDataset(encoded_map)
    #batch_size = len(dataset)     # e.g. 12345 reviews total
    #loader     = DataLoader(
    #    dataset,
    #    batch_size=total_count,   # ← one giant batch
    #    shuffle=False,
    #    collate_fn=collate_fn,
    #    num_workers=8,
    #    pin_memory=True,
    #)

    #with torch.device('cuda'):
    #    for user, values in user_map.items():
    #        encoded_strings = list()
    #        for text in values.reviewTexts:
    #total_count += 1
    #encoded_map[user] = encoded_strings


    #with torch.no_grad():
    #    for batch in loader:                           # ← only one iteration
    #        # 1) Pull out the list of user IDs for this batch
    #        user_ids = batch.pop("user_ids")           # list[str] of length = len(dataset)

    #        # 2) Move tensors to the GPU
    #        batch = {k: v.to(device) for k, v in batch.items()}

    #        # 3) Single forward pass over *all* data
    #        outputs = model(**batch)

    #        # 4) Compute probabilities and bring them back to CPU once
    #        probs       = torch.nn.functional.softmax(outputs.logits, dim=-1)
    #        probs_list  = probs.cpu().tolist()         # List[List[float]] of shape (N_examples, n_classes)

    #        # 5) Re‐associate each example’s probabilities with its user_id
    #        for uid, score_vec in zip(user_ids, probs_list):
    #            personality_map[uid].append(parse_completion(score_vec))

# 6)# Finally, average each user’s list of predictions:
    #final_map = {
    #    uid: personality_average(score_list)
    #    for uid, score_list in personality_map.items()
    #}
    #print(len(final_map)timestamp)
    #print("Finished Inference")

    #cur_count = 0
    #for user_id, encodings in encoded_map:
    #    personality_list = list()
    #    #user, encodings = user_encodings.item()
    #    for encoded in encodings:
    #        with torch.device('cuda'): 
    #        #    encoded = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    #            #encoded_strings.append(encoded)
    #            model.eval() 
    #            with torch.no_grad():
    #                outputs = model(**encoded) #Query model

    #            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    #            predicted_scores = predictions[0].tolist()
    #            personality = parse_completion(predicted_scores)
    #            personality_list.append(personality)
    #    final_personality = personality_average(personality_list)
    #    cur_count += 1
    #    print(f"Infering : {cur_count}/{total_count}")
    #    personality_map[users[user_id]] = final_personality

    #return personality_map

def parse_jsonl(lines):
    user_map = dict()
    movie_map = dict()
    count = 0
    print(lines[0])
    for line in lines:
        json_item = json.loads(line.strip())
        if "text" not in json_item:
            continue
        user_id = json_item["user_id"]
        movie_id = json_item["asin"]
        user = None
        if user_id in user_map:
            user = user_map[user_id]
        else:
            user = Reviewer()
            user.reviewerID = user_id
        user.movies_watched[movie_id] = (json_item["rating"], json_item["timestamp"])
        user.reviewTexts[movie_id] = json_item["text"]
        
        user_set = None
        if movie_id in movie_map:
            user_set = movie_map[movie_id]
        else:
            user_set = set()

        user_set.add(user_id)
        movie_map[movie_id] = user_set
        user_map[user_id] = user

        count += 1
        print(f"{count}/{len(lines)}")


    return (user_map, movie_map)

def cache_20_core(user_map, movie_map):
    with open("/tmp/dataset/20_core_user.jsonl", "w") as file:
        for user_id, user in user_map.items():
            json.dump(user.to_dict(), file)
            file.write('\n')

    with open("/tmp/dataset/20_core_movie.jsonl", "w") as file:
        for movie_id, user_set in movie_map.items():
            json.dump({
                "movieID": movie_id,
                "users": list(map(lambda x: x, user_set)),
                }, file)
            file.write('\n')

def load_20_core():
    user_map = dict()
    with open("/tmp/dataset/20_core_user.jsonl", "r") as file:
        for line in file:
            item = json.loads(line.strip())
            user_id = item["userID"]
            user_map[user_id] = user_from_dict(user_id, item["moviesWatched"], item["reviewTexts"])
    movie_map = dict()
    with open("/tmp/dataset/20_core_movie.jsonl", "r") as file:
        for line in file:
            item = json.loads(line.strip())
            movie_map[item["movieID"]] = set(item["users"])
    return (user_map, movie_map)

def split_dataset(user_map, eval_percentage):
    perc_decimal = eval_percentage / 100

    random.seed(SPLIT_RANDOM_SEED)

    eval_map = dict()

    for user in user_map.values():
        #print(user.reviewTexts.keys())
        eval_user = Reviewer()
        user_id = user.reviewerID
        eval_user.reviewerID = user_id
        movie_count = len(user.movies_watched)
        eval_movie_count = int(movie_count * perc_decimal)
        selected_items = random.sample(list(user.movies_watched.items()), k=eval_movie_count)
        for movie_id, rating in selected_items:
            #print(movie_id)
            eval_user.movies_watched[movie_id] = rating
            eval_user.reviewTexts[movie_id] = user.reviewTexts[movie_id]
            del user.movies_watched[movie_id]
            del user.reviewTexts[movie_id]

        eval_map[user_id] = eval_user

    return (user_map, eval_map)




def main():
    cached_core = False
    user_map = dict()
    movie_map = dict()
    if os.path.exists("/tmp/dataset/20_core_user.jsonl") and os.path.exists("/tmp/dataset/20_core_movie.jsonl"):
        print("loading cached 20-core")
        cached_core = True
        user_map, movie_map = load_20_core()
    else:
        print("loading dataset")
        with open("./Movies_and_TV.jsonl", "r") as file:
            for line in file:
                json_item = json.loads(line.strip())
                if "text" not in json_item:
                    continue
                user_id = json_item["user_id"]
                movie_id = json_item["asin"]
                user = None
                if user_id in user_map:
                    user = user_map[user_id]
                else:
                    user = Reviewer()
                    user.reviewerID = user_id
                user.movies_watched[movie_id] = (json_item["rating"], json_item["timestamp"])
                user.reviewTexts[movie_id] = json_item["text"]
                
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




    if cached_core == False:
        print("20 Core Filtering...")
        while k_core_filter_pass(user_map, movie_map, 20) != False:
            continue
        print("\n\nProcessed 20-core filter passes, user count = " + str(len(user_map)) + "\nmovie count:" + str(len(movie_map)))
        print("caching...")
        cache_20_core(user_map, movie_map)

    print("splitting dataset")
    user_map, eval_map = split_dataset(user_map, 30)

    personality_map = perform_prediction(user_map)

    print("Writing excel to /tmp/dataset")

    training_frame = decompose_movie_map(movie_map, user_map, personality_map)
    eval_frame = decompose_movie_map(movie_map, eval_map, personality_map)

    #user_frame = decompose_user_map(user_map) #, personality_m) work personality matrix in eventually

    #user_dataframe = pd.DataFrame(personality_map, columns=["user_id", "openness", "extraversion", "agreeableness", "conscientiousness", "neuroticism"])
    training_dataframe = pd.DataFrame(training_frame, columns=["movie_id", "user_id", "score", "timestamp", "openness", "extraversion", "agreeableness", "conscientiousness", "neuroticism"])
    eval_dataframe = pd.DataFrame(eval_frame, columns=["movie_id", "user_id", "score", "timestamp", "openness", "extraversion", "agreeableness", "conscientiousness", "neuroticism"])


    with pd.ExcelWriter("/tmp/dataset/out.xlsx") as excel_out:
        #user_dataframe.to_excel(excel_out, sheet_name="User")
        training_dataframe.to_excel(excel_out, sheet_name="training")
        eval_dataframe.to_excel(excel_out, sheet_name="eval")


        

    #average = 0
    #highest = 0
    #lowest = 0
    #for user in user_map.values():
    #    length = len(re.findall(r'\w+', user.reviewTexts))
    #    average += length
    #    if length < lowest or lowest == 0:
    #        lowest = length
    #    if length > highest or highest == 0:
    #        highest = length

    #average = average / len(user_map.keys())

    #print(f"Review Text Word Count Statistics:\naverage : {average}\nhighest : {highest}\nlowest : {lowest}\n")


    #perform_personality_inference(user_map, movie_map)


    input("Press Enter to Continue....")
    return 0

if __name__ == '__main__':
    sys.exit(main())

