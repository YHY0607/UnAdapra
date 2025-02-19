import xml.etree.ElementTree as ET
import json
from tqdm import tqdm
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def parse_posts_xml(r_path, w_path):
    # From  Posts.xml to Posts.json
    lines = open(r_path, 'r').readlines()
    Posts = {}
    IndexError_cnt = 0

    for i in tqdm(range(len(lines))):
        if i < 2 or i == len(lines) - 1:
            continue

        root = ET.fromstring(lines[i])
        post_id = root.get('Id')
        post_type = root.get('PostTypeId')

        if post_type == '1':
            if root.get('AcceptedAnswerId')=='null':
                continue
            Post = {
                'body': root.get('Body'),
                'title': root.get('Title'),
                'answer': root.get('AcceptedAnswerId'),
                'score': root.get('Score'),
                'tags': root.get('Tags'),
                'time': root.get('CreationDate'),
                'answers': []
            }
            Posts[post_id] = Post

        elif post_type == '2':
            parent_id = root.get('ParentId')
            if parent_id in Posts:
                Answer = {
                    'post_id': post_id,
                    'body': root.get('Body'),
                    'score': root.get('Score'),
                    'tags': root.get('Tags'),
                    'time': root.get('CreationDate')
                }
                Posts[parent_id]['answers'].append(Answer)
            else:
                IndexError_cnt += 1

    print('IndexError_cnt:', IndexError_cnt)
    
    with open(w_path, 'w') as fw:
        for post_id, post in Posts.items():  
            fw.write(json.dumps({post_id: post}))
            fw.write('\n')
    fw.close()
def choose_accepted(r_path,w_path):
    lines=open(r_path,'r').readlines()
    fw= open(w_path, 'w') 
    for line in tqdm(lines):
        line=json.loads(line.strip())
        post=list(line.values())[0]
        if post["answer"]==None:
            continue
        fw.write(json.dumps(line))
        fw.write('\n')
    fw.close()  
def choose_code(r_path,w_path):
    # from Post.json to Posts_qcode.json
    lines=open(r_path,'r').readlines()
    fw= open(w_path, 'w') 
    for line in tqdm(lines):
        line=json.loads(line.strip())
        post_id= list(line.keys())[0]
        post=list(line.values())[0]
        if "<code>" in line[post_id]['body'] or "<code>" in line[post_id]['title']:
            fw.write(json.dumps({post_id: post}))
            fw.write('\n')
    fw.close() 
import numpy as np   
# def choose_good_test_data(r_path,w_path):
#     lines=open(r_path,'r').readlines()
#     # fw= open(w_path, 'w') 
#     title_cnt=[]
#     body_cnt=[]
#     t_body_cnt=[]
#     answers_cnt=[]
#     for line in tqdm(lines):
#         line=json.loads(line.strip())
#         post_id= list(line.keys())[0]
#         post=list(line.values())[0]
#         answers=post["answers"]
#         sorted_answers = sorted(answers, key=lambda x: int(x["score"]), reverse=True)
#         ans_lens=[]
#         if len(answers)<5:
#             continue
#         for answer in answers:
#             answer_body=answer["body"].split(' ')
#             ans_lens.append(len(answer_body))
#         if np.mean(ans_lens[:5])>256:
#             continue
#         answers_cnt.append(np.mean(ans_lens[:5]))
#         title=post["title"].split(' ')
#         title_cnt.append(len(title))
#         body=post["body"].split(' ')
#         body_cnt.append(len(body))
#         t_body_cnt.append(len(title)+len(body))
        
        
#     print("title_cnt: ", np.mean(title_cnt))
#     print("body_cnt: ", np.mean(body_cnt))
#     print("t_body_cnt: ", np.mean(t_body_cnt))
#     print("answers_cnt: ", np.mean(answers_cnt))
#     print(len(title_cnt))
   
    # fw.close()     
# choose_good_test_data('/data/model_and_dataset/Posts_ac_qcode.json','')
def choose_cnt(r_path,w_path):
    lines=open(r_path,'r').readlines()
    fw= open(w_path, 'w') 
    five_cnt=0
    for line in tqdm(lines):
        line=json.loads(line.strip())
        post_id= list(line.keys())[0]
        post=list(line.values())[0]
        answers=post["answers"]
        sorted_answers = sorted(answers, key=lambda x: int(x["score"]), reverse=True)
        ans_lens=[]
        answer_cnt=0
        scores=[]
        tags=post["tags"]
        # if 'python' or 'java' not in tags:
        #     continue
        if len(answers)<5:
            continue
        
        for answer in sorted_answers[:5]:
            
            # answer_body=answer["body"].split(' ')
            # if len(answer_body)>64:
            #     continue
            if int(answer['score'])<1:
                continue
          
        fw.write(json.dumps({post_id:post})+'\n')
        five_cnt+=1
    print("five_cnt: ",five_cnt)
def choose_good_test_data(r_path,w_path):
    lines=open(r_path,'r').readlines()
    fw= open(w_path, 'w') 
    title_cnt=[]
    body_cnt=[]
    t_body_cnt=[]
    answers_cnt=[]
    for line in tqdm(lines):
        line=json.loads(line.strip())
        post_id= list(line.keys())[0]
        post=list(line.values())[0]
        answers=post["answers"]
        sorted_answers = sorted(answers, key=lambda x: int(x["score"]), reverse=True)
        ans_lens=[]
        answer_cnt=0
        scores=[]
        tags=post["tags"]
        new_answers=[]
        for answer in sorted_answers[:5]:
            if int(answer['score'])<1:
                continue
            answer_body=answer["body"].split(' ')
            if len(answer_body)>128:
                continue
            
            ans_lens.append(len(answer_body))
            scores.append(int(answer["score"]))
            answer_cnt+=1
            new_answers.append(answer)
        if answer_cnt<5:
            continue
        if max(scores)-min(scores)<200:
            continue
        answers_cnt.append(np.mean(ans_lens))
        title=post["title"].split(' ')
        title_cnt.append(len(title))
        body=post["body"].split(' ')
        if len(title)+len(body)>64:
            continue
        body_cnt.append(len(body))
        t_body_cnt.append(len(title)+len(body))
        post['answers']=new_answers
        fw.write(json.dumps({post_id:post})+'\n')
    print(r_path)    
    print("title_cnt: ", np.mean(title_cnt))
    print("body_cnt: ", np.mean(body_cnt))
    print("t_body_cnt: ", np.mean(t_body_cnt))
    print("answers_cnt: ", np.mean(answers_cnt))
    print(len(title_cnt))

def choose_good_train_data(r_path,w_path):
    lines=open(r_path,'r').readlines()
    fw= open(w_path, 'w') 
    title_cnt=[]
    body_cnt=[]
    t_body_cnt=[]
    answers_cnt=[]
    for line in tqdm(lines):
        line=json.loads(line.strip())
        post_id= list(line.keys())[0]
        post=list(line.values())[0]
        answers=post["answers"]
        sorted_answers = sorted(answers, key=lambda x: int(x["score"]), reverse=True)
        ans_lens=[]
        answer_cnt=0
        scores=[]
        tags=post["tags"]
        new_answers=[]
        for answer in answers:
            
            answer_body=answer["body"].split(' ')
            if int(answer['score'])<1:
                continue
            ans_lens.append(len(answer_body))
            scores.append(int(answer["score"]))
            answer_cnt+=1
            new_answers.append(answer)
        if answer_cnt<3:
            continue
        if max(scores)-min(scores)<10:  # 通过分数，字符长度控制训练集大小
            continue
        answers_cnt.append(np.mean(ans_lens))
        title=post["title"].split(' ')
        title_cnt.append(len(title))
        body=post["body"].split(' ')
        
        body_cnt.append(len(body))
        t_body_cnt.append(len(title)+len(body))
        post['answers']=new_answers
        fw.write(json.dumps({post_id:post})+'\n')
        
    print("title_cnt: ", np.mean(title_cnt))
    print("body_cnt: ", np.mean(body_cnt))
    print("t_body_cnt: ", np.mean(t_body_cnt))
    print("answers_cnt: ", np.mean(answers_cnt))
    print(len(title_cnt))
def choose_gpt_data(r_path,w_path):
    lines=open(r_path,'r').readlines()
    fw= open(w_path, 'w') 
    title_cnt=[]
    body_cnt=[]
    t_body_cnt=[]
    answers_cnt=[]
    for line in tqdm(lines):
        line=json.loads(line.strip())
        post_id= list(line.keys())[0]
        post=list(line.values())[0]
        time=post['time']
        now_obj = datetime.now()
        diff_year=(now_obj-datetime.fromisoformat(time)).days/365
        if diff_year>10:
            continue
        fw.write(json.dumps({post_id:post})+'\n')
        
   
    # fw.close()     

def check_title(r_path,w_path):
    line=open(r_path,'r').readlines()
    fw=open(w_path,'w')
    for l in tqdm(line):
        l=json.loads(l.strip())
        post_id= list(l.keys())[0]
        post=list(l.values())[0]
        title=post["title"]+": "+post["body"]
        
        fw.write(json.dumps({'title':title,'time':post['time']})+'\n')
def check_infer(r_path,w_path):
    line=open(r_path,'r').readlines()
    fw=open(w_path,'w')
    for l in tqdm(line):
        l=json.loads(l.strip())
        post_id= list(l.keys())[0]
        post=list(l.values())[0]
        title=post["title"]+": "+post["body"]
        
        fw.write(json.dumps({'title':title,'time':post['time']})+'\n')
def choose_time(r_path,w_path):
    # generate x years and y length data : 2-year_5-cnt.json
    lines=open(r_path,'r').readlines()
    print(lines[0])
    now = datetime.now()
    two_years_ago = now - relativedelta(years=2)
    four_years_ago = now - relativedelta(years=4)
    six_years_ago = now - relativedelta(years=6)
    eight_years_ago = now - relativedelta(years=8)
    filtered_data_2 = []
    filtered_data_4 = []
    filtered_data_6 = []
    filtered_data_8 = []
    for line in tqdm(lines):
        line=json.loads(line.strip())
        post_id= list(line.keys())[0]
        # if datetime.fromisoformat(line[post_id]["time"]) > two_years_ago and len(line[post_id]["answers"])>=5:
        if len(line[post_id]["answers"])<8 and len(line[post_id]["answers"])>=4:

            filtered_data_2.append(line)
    print("w_path cnt: ",len(filtered_data_2))
    with open(w_path,'w') as fw:
        for line in tqdm(filtered_data_2):
            fw.write(json.dumps(line)+'\n')
    #     if datetime.fromisoformat(line[post_id]["time"]) > four_years_ago:
    #         filtered_data_4.append(line)
    #     if datetime.fromisoformat(line[post_id]["time"]) > six_years_ago:
    #         filtered_data_6.append(line)
    #     if datetime.fromisoformat(line[post_id]["time"]) > eight_years_ago:
    #         filtered_data_8.append(line)
    # cnt(filtered_data_2)
    # cnt(filtered_data_4)
    # cnt(filtered_data_6)
    # cnt(filtered_data_8)
def cnt(lines):
   
    cnt_3,cnt_5,cnt_8,cnt_10,cnt_15,cnt_20=0,0,0,0,0,0
    for line in tqdm(lines):
        # line=json.loads(line.strip())
        if len(list(line.values())[0]['answers'])>=3:
            cnt_3+=1
        if len(list(line.values())[0]['answers'])>=5:
            cnt_5+=1
        if len(list(line.values())[0]['answers'])>=8:
            cnt_8+=1
        if len(list(line.values())[0]['answers'])>=10:
            cnt_10+=1
        if len(list(line.values())[0]['answers'])>=15:
            cnt_15+=1
        if len(list(line.values())[0]['answers'])>=20:
            cnt_20+=1
    print('cnt_3:  ',cnt_3)
    print('cnt_5:  ',cnt_5)
    print('cnt_8:  ',cnt_8)
    print('cnt_10:  ',cnt_10)
    print('cnt_15:  ',cnt_15)
    print('cnt_20:  ',cnt_20)
 
from bs4 import BeautifulSoup


def remove_html_except_code(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    for tag in soup.find_all(True):
        if tag.name != 'code':
            tag.unwrap()
    return str(soup)
   
def deal_html():
    html_text = '''
     Change the service account to an AD account and register the SPN's as shown. Use your own service name e.g. fooservice\n\nsetspn -A fooservice/servermachinename domain\\serviceAccountName\n  setspn -A fooservice/servermachinename.fullyqualifieddomainname\n  domain\\serviceAccountName\n\nIn the client config set:\n<code>&lt;identity&gt;\n    &lt;serviceprincipalname value=\"fooservice/servermachinename\" /&gt;\n&lt;/identity&gt;\n</code>\n
    '''
    soup = BeautifulSoup(html_text, 'html.parser')
    plain_text = soup.get_text()
    print(plain_text)
def clean(r_path,w_path):
    datas=open(r_path,'r').readlines()
    fw=open(w_path,'w')
    for data in tqdm(datas):
        find=False
        data=json.loads(data.strip())
        post_id = list(data.keys())[0]
        post=data[post_id]
        post["body"]=remove_html_except_code(post["body"])
        post["title"]=remove_html_except_code(post["title"])
        answer_id=post["answer"]
        new_answers=[]
        for i, answer in enumerate(post["answers"], start=1):
            if int(answer['score'])<1:
                continue
            answer["answer_id"] = i
            answer["body"]=remove_html_except_code(answer["body"])
            if answer["post_id"] == answer_id:
                find=True
                post['answer_body']=answer["body"]
                post["answer_number"]=i
            new_answers.append(answer)
        post["answers"]=new_answers
        if find:
            fw.write(json.dumps({post_id:post})+'\n')
    fw.close()
def look(r_path):
    lines=open(r_path).readlines()
    print(len(lines))
    print(lines[0])


from datetime import datetime
    
def time_norm(r_path,w_path):
    score=11
    query_time_obj = datetime.strptime('2022-07-24T15:06:05.230', "%Y-%m-%dT%H:%M:%S.%f")
    answer_time_obj= datetime.strptime('2022-10-22T18:59:16.703', "%Y-%m-%dT%H:%M:%S.%f")
    difference_time= (answer_time_obj-query_time_obj).total_seconds()
    norm_vote=score/difference_time
    print(query_time_obj)
    print(query_time_obj.timestamp)
    print(difference_time)
    print(norm_vote)
    
def difference_time_decay(r_path,w_path):
    score = 11
    query_time_str = '2022-07-24T15:06:05.230'
    answer_time_str = '2022-10-22T18:59:16.703'
    
    # 解析时间字符串
    query_time_obj = datetime.strptime(query_time_str, "%Y-%m-%dT%H:%M:%S.%f")
    answer_time_obj = datetime.strptime(answer_time_str, "%Y-%m-%dT%H:%M:%S.%f")
    
    # 转换为时间戳
    query_timestamp = query_time_obj.timestamp()
    answer_timestamp = answer_time_obj.timestamp()
    
    # 计算时间差（以秒为单位）
    time_difference = answer_timestamp - query_timestamp
    
    # 计算时间衰减
    decay_factor = 1 / (1 + time_difference)
    decayed_score = score * decay_factor
    
    print("Query time object:", query_time_obj)
    print("Query timestamp:", query_timestamp)
    print("Answer time object:", answer_time_obj)
    print("Answer timestamp:", answer_timestamp)
    print("Decayed score:", decayed_score)
from collections import defaultdict
def tags_cnt(r_path,w_path):
    lines=open(r_path).readlines()
    tags_cnt=defaultdict(int)
    for line in tqdm(lines):
        line=json.loads(line.strip())
        post=list(line.values())[0]
        tags=post["tags"].split("|")
        for tag in tags:
            tags_cnt[tag]+=1
    sorted_tag_count = dict(sorted(tags_cnt.items(), key=lambda x: x[1], reverse=True))
    fw=open(w_path,'w')
    for tag, count in sorted_tag_count.items():
        fw.write(json.dumps({tag: [count,count/len(lines)]})+'\n')
    fw.close()

import pandas as pd

def ana(r_path):
    lines=open(r_path).readlines()
    votes=[]
    for line in tqdm(lines):
        line=json.loads(line.strip())
        post=list(line.values())[0]
        answers=post["answers"]
        for answer in answers:
           vote=int(answer["score"])
           votes.append(vote)
    print("max vote: ",np.max(votes))
    print("min vote: ",np.min(votes))
    print("mean vote: ",np.mean(votes))
    print("std vote: ",np.std(votes))
    print("median vote: ",np.median(votes))
    print("vote 0 cnt: ",len([vote for vote in votes if vote==0]))
    print("vote neg cnt: ",len([vote for vote in votes if vote<0]))
import random
def choose_five_answer(r_path,w_path):
    lines=open(r_path).readlines()
    fw=open(w_path,'w')
    for line in tqdm(lines):
        line=json.loads(line.strip())
        post_id= list(line.keys())[0]
        post=list(line.values())[0]
        answers=post["answers"]
        specified_answer_id=post['answer_number']
        selected_answers=[]
        for answer in answers:
            if answer["answer_id"] == specified_answer_id:
                selected_answers.append(answer)

        # 2. 按照 score 排序并选择前两个
        sorted_by_score = sorted(answers, key=lambda x: int(x["score"]), reverse=True)
        top_two_by_score = [answer for answer in sorted_by_score if answer not in selected_answers][:2]
        selected_answers.extend(top_two_by_score)

        # 3. 按照时间排序选择最新的两个答案
        sorted_by_time = sorted(answers, key=lambda x: datetime.fromisoformat(x["time"]), reverse=True)

        latest_answers = []
        for answer in sorted_by_time:
            if answer not in selected_answers and (len(latest_answers) < 2):
                if int(answer["score"]) > 0 or len(latest_answers) > 0:  # 确保至少有两个答案
                   latest_answers.append(answer)

        selected_answers.extend(latest_answers)
        random.shuffle(selected_answers)
        post['answers']=selected_answers
        fw.write(json.dumps({post_id:post})+'\n')
    fw.close()
# choose_five_answer("/home/yanghongyu/codeqa/data/gpt1.json","/home/yanghongyu/codeqa/data/gpt1_five.json")  
# choose_cnt('/data/model_and_dataset/Posts_ac_qcode.json','/home/yanghongyu/codeqa/data/5-cnt.json')   
# clean('/home/yanghongyu/codeqa/data/5-cnt.json','/home/yanghongyu/codeqa/data/5-cnt_clean.json') 

# choose_good_test_data('/home/yanghongyu/codeqa/data/5-cnt.json','/home/yanghongyu/codeqa/data/test.json')
# clean('/home/yanghongyu/codeqa/data/test.json','/home/yanghongyu/codeqa/data/test_clean.json')
# choose_gpt_data('/home/yanghongyu/codeqa/data/test_clean.json','/home/yanghongyu/codeqa/data/gpt.json')
# check_title('/home/yanghongyu/codeqa/data/gpt.json','/home/yanghongyu/codeqa/data/title.json')
# choose_five_answer('/home/yanghongyu/codeqa/data/gpt.json','/home/yanghongyu/codeqa/data/gpt_five.json')
# ana('/home/yanghongyu/codeqa/data/gpt_five.json')
choose_good_train_data('/home/yanghongyu/codeqa/data/5-cnt.json','/home/yanghongyu/codeqa/data/new_train.json')
clean('/home/yanghongyu/codeqa/data/new_train.json','/home/yanghongyu/codeqa/data/new_train_clean.json')
# choose_five_answer('/home/yanghongyu/codeqa/data/new_train_clean.json','/home/yanghongyu/codeqa/data/new_train_five.json')
# ana('/home/yanghongyu/codeqa/data/new_train_five.json')
# clean('/home/yanghongyu/codeqa/data/new_train.json','/home/yanghongyu/codeqa/data/new_train_clean.json')
# choose_five_answer('/home/yanghongyu/codeqa/data/new_train_clean.json','/home/yanghongyu/codeqa/data/new_train_five.json')
# ana('/home/yanghongyu/codeqa/data/new_train_five.json')
# clean('/home/yanghongyu/codeqa/data/new_dataset_1.json','/home/yanghongyu/codeqa/data/new_dataset_clean_1.json')
# choose_gpt_data('/home/yanghongyu/codeqa/data/new_dataset_clean_1.json','/home/yanghongyu/codeqa/data/gpt1.json')
# # clean('/home/yanghongyu/codeqa/data/gpt.json','/home/yanghongyu/codeqa/data/gpt_clean.json')
# check_title('/home/yanghongyu/codeqa/data/gpt1.json','/home/yanghongyu/codeqa/data/title1.json')
# ana('/home/yanghongyu/codeqa/data/new_dataset_clean.json')
# deal_sa("/home/yanghongyu/codeqa/data/public/data/Stackoverflow.com/train_ori.json",3) #58723
# deal_sa("/home/yanghongyu/codeqa/data/public/data/Stackoverflow.com/train_ori.json",5)

# deal_public("/home/yanghongyu/codeqa/data/public/data/chemistry.stackexchange.com/train-00000-of-00001.parquet","/home/yanghongyu/codeqa/data/public/data/chemistry.stackexchange.com/train.json") 
# deal_public("/home/yanghongyu/codeqa/data/public/data/chemistry.stackexchange.com/Stackoverflow.com/train-00000-of-00335.parquet","/home/yanghongyu/codeqa/data/public/data/chemistry.stackexchange.com/Stackoverflow.com/train1.json") 
# deal_public("/home/yanghongyu/codeqa/data/public/data/chemistry.stackexchange.com/Stackoverflow.com/train-00001-of-00335.parquet","/home/yanghongyu/codeqa/data/public/data/chemistry.stackexchange.com/Stackoverflow.com/train2.json") 
# deal_public("/home/yanghongyu/codeqa/data/public/data/chemistry.stackexchange.com/Stackoverflow.com/train-00002-of-00335.parquet","/home/yanghongyu/codeqa/data/public/data/chemistry.stackexchange.com/Stackoverflow.com/train3.json") 

# tags_cnt("/data/model_and_dataset/Posts_ac_qcode.json","/home/yanghongyu/codeqa/data/tags.json")
# def generate_new_test_dataset(r_path,w_path):
#     lines=open(r_path).readlines()
        # print(f"{tag}: {count}")
# time_norm("","")  #1.4120682979083923e-06
# difference_time_decay(" "," ")
# print(null==null)

# look("/data/model_and_dataset/Posts_ac.json")
# lines = open("/data/model_and_dataset/Posts_ac.json").readlines()
# # lines  = json.load(open("/data/model_and_dataset/Posts_ac.json", "r"))
# cnt(lines)
# choose_code('/data/model_and_dataset/Posts_ac.json','/data/model_and_dataset/Posts_ac_qcode.json')
# deal_html()
# choose_accepted('/data/model_and_dataset/Posts.json',"/data/model_and_dataset/Posts_ac_1.json")
# choose_code('/data/model_and_dataset/Posts.json','/data/model_and_dataset/Posts_qcode.json')
# choose_time('/data/model_and_dataset/Posts_ac_qcode.json','/home/yanghongyu/codeqa/data/2-year_5-cnt.json')
# choose_time('/data/model_and_dataset/Posts_ac_qcode.json','/home/yanghongyu/codeqa/data/ge4-l8-cnt.json')
# clean('/home/yanghongyu/codeqa/data/8-cnt.json','/home/yanghongyu/codeqa/data/8-cnt_clean.json')
# choose_code('/data/model_and_dataset/Posts.json','')
# cnt( '/data/model_and_dataset/Posts.json')
# 示例调用
# parse_posts_xml('/data/model_and_dataset/Posts.xml', '/data/model_and_dataset/Posts.json')


# 原始JSON数据
# data = {
#     "38779": {
#         "body": "<p>I have a wcf application hosted in a windows service running a local windows account. Do I need to set an SPN for this account? If so, what's the protocol the SPN needs to be set under? I know how to do this for services over HTTP, but have never done it for net.tcp.</p>\n",
#         "title": "What SPN do I need to set for a net.tcp service?",
#         "answer": "40472",
#         "score": "6",
#         "tags": "|wcf|security|spn|",
#         "time": "2008-09-02T03:41:06.880",
#         "answers": [
#             {
#                 "post_id": "40472",
#                 "body": "<p>By default (i.e. out of the box) net.tcp services are unsecured and don't perform any authentication at all. So you won't need (and in fact can't) set a service principal name. </p>\n\n<p>If you need to authenticate, then check the <a href=\"http://msdn.microsoft.com/en-us/library/system.servicemodel.nettcpsecurity.aspx\" rel=\"nofollow noreferrer\">net.tcp security</a> modes on MSDN. The best way to understand the different combinations is to experiment!</p>\n",
#                 "score": "3",
#                 "tags": None,
#                 "time": "2008-09-02T20:12:02.473"
#             },
#             {
#                 "post_id": "71039",
#                 "body": "<p>Change the service account to an AD account and register the SPN's as shown. Use your own service name e.g. fooservice</p>\n\n<blockquote>\n  <p>setspn -A fooservice/servermachinename domain\\serviceAccountName<br>\n  setspn -A fooservice/servermachinename.fullyqualifieddomainname\n  domain\\serviceAccountName</p>\n</blockquote>\n\n<p>In the client config set:</p>\n\n<pre><code>&lt;identity&gt;\n    &lt;serviceprincipalname value=\"fooservice/servermachinename\" /&gt;\n&lt;/identity&gt;\n</code></pre>\n",
#                 "score": "6",
#                 "tags": None,
#                 "time": "2008-09-16T10:29:43.360"
#             }
#         ]
#     }
# }

# # 解析数据
# question_id = "38779"
# question_data = data[question_id]
# data[question_id]['body']=remove_html_except_code(data[question_id]['body'])
# data[question_id]['title']=remove_html_except_code(data[question_id]['title'])

# # 找到answer对应的body
# answer_id = question_data["answer"]

# for answer in question_data["answers"]:
#     if answer["post_id"] == answer_id:
#         data[question_id]['answer_body'] = answer["body"]
#         break

# # 给answers里的每个answer编号
# for i, answer in enumerate(question_data["answers"], start=1):
#     answer["number"] = i
#     answer["body"]=remove_html_except_code(answer["body"])

# # 根据score进行排序，生成vote_order
# sorted_answers = sorted(question_data["answers"], key=lambda x: int(x["score"]), reverse=True)
# vote_order = [answer["number"] for answer in sorted_answers]

# # 添加新的键 'vote_order'
# question_data["vote_order"] = vote_order

# # 输出结果
# print(json.dumps(data, indent=4))
