
import requests as r
import re


# param is either "1000_common_words.txt" or "common_words.txt"
def get_common_words(file_name):
    a_file = open(file_name, "r")
    common_words = []
    for line in a_file:
        stripped_line = line.strip()
        common_words.append(stripped_line)
    a_file.close()
    return(common_words)

def get_charged_words(num_words=100):
    # url ="https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-111.txt"
    a_file = open("AFINN-111.txt", "r")
    common_words = []
    word_to_score={}
    valid_cnt=0
    error_cnt=0
    for line in a_file:
        stripped_line = line.strip()
        score = re.findall('[-0-9]+', line)
        score = str(score[0])
        line_split = line.split()
        word= line_split[0] 
        score_2 = line_split[1]
        score_2=str(score_2)
        
        if score_2==score:
            word_to_score[word]=score
            valid_cnt+=1
        else:
            error_cnt+=1
        # print(score_2)
    num_words=int(num_words/2)
    biggest_items = sorted(word_to_score.items(), reverse=True)[0:num_words]
    smallest_items=sorted(word_to_score.items())[0:num_words]
    charged_words = [x[0] for x in biggest_items] + [x[0] for x in smallest_items]
    return charged_words

