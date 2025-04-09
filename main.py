import json
import sys

class Reviewer:
    def __init__(self):
        self.reviewerID = None
        self.movies_watched = dict()
        self.reviewTexts = ""

def ten_core_filter_pass(user_map, movie_map):
    pending_removal = list()
    filtered = False
    for movie_id in movie_map.keys():
        movie = movie_map[movie_id]
        if len(movie) < 10:
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
        if len(user.movies_watched) < 10:
            filtered = True
            pending_removal.append(user.reviewerID)
    for user_id in pending_removal:
        user = user_map[user_id]
        for movie_id in user.movies_watched:
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
        while ten_core_filter_pass(user_map, movie_map) != False:
            continue


        print("\n\n Processed 10-core filter passes, user count = " + str(len(user_map)) + "\nmovie count:" + str(len(movie_map)))

        input("Press Enter to Continue....")
        return 0

if __name__ == '__main__':
    sys.exit(main())

