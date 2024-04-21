from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from datetime import datetime
import re
import json
from tqdm import tqdm

def event_extra(event_output):
    event_pattern = r"time\s*:\s*\"datetime\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)\s*-\s*datetime\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)\"\s*,\s*object\s*:\s*\"([^\"]+)\""
    # event_matches = []
    time_ranges = []
    # for events_ in event_output:
    event_matches = re.findall(event_pattern, event_output)
    for match in event_matches:
        if 0<int(match[0])<2024 and 0<int(match[1])<=12 and 0<int(match[2])<=31:
            start_date = datetime(int(match[0]), int(match[1]), int(match[2]))
        elif 0<int(match[2])<2024 and 0<int(match[1])<=12 and 0<int(match[0])<=31:
            start_date = datetime(int(match[2]), int(match[1]), int(match[0]))
        else:
            continue

        if 0<int(match[3])<2024 and 0<int(match[4])<=12 and 0<int(match[5])<=31:
            end_date = datetime(int(match[3]), int(match[4]), int(match[5]))
        elif 0<int(match[5])<2024 and 0<int(match[4])<=12 and 0<int(match[3])<=31:
            end_date = datetime(int(match[5]), int(match[4]), int(match[3]))   
        else:
            continue

        description = match[6]
        time_ranges.append((start_date, end_date, description))
    
    if len(time_ranges) == 0:
        event_pattern2 = r"time\s*:\s*\"([^\"]+)\"\s*,\s*object\s*:\s*\"([^\"]+)\""
        event_matches2 = re.findall(event_pattern2, event_output)
        for match2 in event_matches2:
            dash_count = match2[0].count("-")
            if dash_count !=1:
                continue
            time_str = match2[0].split('-')
            time_list = []
            for t, time in enumerate(time_str):
                numbers_list = [[1, 1, 1],[2023,1,1]]
                numbers = re.findall(r'\d+', time)
                for i, number in enumerate(numbers):
                    numbers_list[t][i]=int(number)
                if 0<numbers_list[t][0]<2024 and 0<numbers_list[t][1]<=12 and 0<numbers_list[t][2]<=31:
                    time_ = datetime(int(numbers_list[t][0]), int(numbers_list[t][1]), int(numbers_list[t][2]))
                else:
                    continue
                time_list.append(time_)
            if len(time_list)==2:
                time_ranges.append([time_list[0], time_list[1], match2[1]])
        for i,timeline in enumerate(time_ranges):
            time1 = datetime(int(1), int(1), int(1))
            time2 = datetime(int(2023), int(8), int(6))
            if i>0 and timeline[0]==time1:
                time_ranges[i][0]=time_ranges[i-1][1]
            if i<len(time_ranges)-1 and timeline[1]==time2:
                time_ranges[i][1]=time_ranges[i+1][0]
    return time_ranges

def predict(start_q, end_q, eventline):
    # start_q = datetime.strptime(time[0], '%Y-%m-%d %H:%M:%S')
    # end_q = datetime.strptime(time[1], '%Y-%m-%d %H:%M:%S')
    answer = []
    time_ranges = sorted(eventline, key=lambda x: x[0])
    for event_start, event_end, event_label in time_ranges:
        # overlap = min(end_date_q, event_end) - max(start_date_q , event_start)
        if event_start.year <= start_q.year and  end_q.year<= event_end.year:
            if event_label not in answer:
                answer.append(event_label)
    return answer

def predict2(rel, time, eventline):
    tmie_q = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    # end_q = datetime.strptime(time[1], '%Y-%m-%d %H:%M:%S')
    time_ranges = sorted(eventline, key=lambda x: x[0])
    for event_start, event_end, event_label in time_ranges:
        # overlap = min(end_date_q, event_end) - max(start_date_q , event_start)
        if rel == 'in':
            if event_start.year <= tmie_q.year<= event_end.year:
                return  event_label
        if rel == 'before':
            if tmie_q.year<= event_start.year:
                return  event_label
        if rel == 'after':
            if event_end.year <= tmie_q.year:
                return  event_label
    return None


if __name__ == "__main__":

    with open('dataset/test_easy_multi_Done.json', 'r') as input_file:
        datas = json.load(input_file)
    # datas = datas[0:1]
    with open('chatgpt_output/easy_multi_question.json', 'r') as input_file:
        question_list = json.load(input_file)
    question_json = {}
    for q in question_list:
        question_json[q['idx']]=q['question'][0]

    with open('chatgpt_output/all_eventline.json', 'r') as input_file:
        fact_json = json.load(input_file)
    
    chatgpt_output = []
    for data in tqdm(datas):
        idx = data['idx']
        parts = idx.split('#')  # 将字符串按 '_' 分割成多个部分
        id = '#'.join(parts[:2])  # 取前三个部分并用 '_' 连接起来
        if id in fact_json:
            # answer = {}
            question_output = question_json[idx]
            relation_pattern = r"\"time\"\s*:\s*\"datetime\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)\s*-\s*datetime\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)\""
            event_times = re.findall(relation_pattern, question_output)
            for match in event_times:
                if 0<int(match[0])<2024 and 0<int(match[1])<=12 and 0<int(match[2])<=31:
                    start_date = datetime(int(match[0]), int(match[1]), int(match[2]))
                elif 0<int(match[2])<2024 and 0<int(match[1])<=12 and 0<int(match[0])<=31:
                    start_date = datetime(int(match[2]), int(match[1]), int(match[0]))

                if 0<int(match[3])<2024 and 0<int(match[4])<=12 and 0<int(match[5])<=31:
                    end_date = datetime(int(match[3]), int(match[4]), int(match[5]))
                elif 0<int(match[5])<2024 and 0<int(match[4])<=12 and 0<int(match[3])<=31:
                    end_date = datetime(int(match[5]), int(match[4]), int(match[3]))
                
            facts = fact_json[id]
            event_line = []
            for fact in facts:
                event_line += event_extra(fact)
            predict_answer = predict(start_date, end_date, event_line)

            if len(predict_answer)==0:
                predict_answer = ['none']

            answer_list={"idx":idx, "targets": data['targets'], "prediction_text":predict_answer }

            chatgpt_output.append(answer_list)
        else:
            print(idx)

    
    # 将字典保存为JSON文件
    with open('predict/easy_multi.json', 'w') as outfile:
        json.dump(chatgpt_output, outfile)
    