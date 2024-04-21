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
            break

        if 0<int(match[3])<2024 and 0<int(match[4])<=12 and 0<int(match[5])<=31:
            end_date = datetime(int(match[3]), int(match[4]), int(match[5]))
        elif 0<int(match[5])<2024 and 0<int(match[4])<=12 and 0<int(match[3])<=31:
            end_date = datetime(int(match[5]), int(match[4]), int(match[3]))   
        else:
            break

        description = match[6]
        time_ranges.append((start_date, end_date, description))
    return time_ranges

def predict(time, eventline):
    start_q = datetime.strptime(time[0], '%Y-%m-%d %H:%M:%S')
    end_q = datetime.strptime(time[1], '%Y-%m-%d %H:%M:%S')
    time_ranges = sorted(eventline, key=lambda x: x[0])
    for event_start, event_end, event_label in time_ranges:
        # overlap = min(end_date_q, event_end) - max(start_date_q , event_start)
        if event_start.year <= start_q.year and  end_q.year<= event_end.year:
            return  event_label
    return None


if __name__ == "__main__":
    # string2list()

    llm = ChatOpenAI(model_name = "gpt-3.5-turbo")

    event_template = """
    Instruction: Extract and summarize event timeline about a given relation from the context. An example is as follows.
    Relation: play for team
    Context: Chorley started his career at Arsenal, spending several years at the club on youth terms before signing a professional deal in 2001. Having made no first-team appearances for Arsenal, Chorley opted to leave the club and sign for Wimbledon in 2003. He continued playing under the club's new guise in Milton Keynes for a further three years from 2004 to 2007. During his final year at MK Dons, Chorley spent the majority of the 2006–07 season on loan at Gillingham. He left MK Dons in July 2007, and signed a two-year contract with Tranmere Rovers. 
    extracted_info: 
    Chorley played for Arsenal from 2001 to 2003.
    Chorley played for Wimbledon from 2003 to 2004. 
    Chorley played for Milton Keynes from 2004 to 2007.
    Chorley was on loan at Gillingham from 2006 to 2007.
    Chorley played for Tranmere Rovers from July 2007 to 2009.

    Relation: {Relation}
    Context: {Context}
    extracted_info:
    """
    prompt_template = PromptTemplate(
        input_variables=["Relation","Context"],
        template=event_template,
        )
    chain = LLMChain(llm=llm, prompt=prompt_template)

    chatgpt_output = []

    with open('dataset/all_test.json', 'r') as input_file:
        datas = json.load(input_file)
    datas = datas[10:]
    with open('chatgpt_output/easy_single_question.json', 'r') as input_file:
        quesions_info = json.load(input_file)

    question_json = {}
    for id_data in quesions_info:
        question_output = id_data['question'][0]
        relation_pattern = r"\"relation\"\s*:\s*\"([^\"]+)\""
        relation_ = re.findall(relation_pattern, question_output)
        relation_ = relation_[0]
        idx = id_data['idx']
        parts = idx.split('#')  # 将字符串按 '_' 分割成多个部分
        id = '#'.join(parts[:2])  # 取前三个部分并用 '_' 连接起来
        if id in question_json:
            question_json[id].append(relation_)
        else:
            question_json[id]=[relation_]

    for data in tqdm(datas):
        idx = data['idx']
        # if idx in question_json:
        if idx is not None:
            answer = {}
            facts = data['paragraphs']
            context = ' '.join(item['text'] for item in facts)
            relation = question_json[idx][0]
            event = []
            input_tokens = context.split()
            for i in range(0, len(input_tokens), 2500):
                partial_input = " ".join(input_tokens[i:i+2500])
                output = chain.run({'Relation':relation,'Context':partial_input})
                event.append(output)

            answer_list={}            
            answer_list['idx']=idx
            answer_list['targets'] = data['targets']
            answer_list['event'] = event
            

            chatgpt_output.append(answer_list)

    
    # 将字典保存为JSON文件
    with open('baseline/all_test_event.json', 'a') as outfile:
        json.dump(answer_list, outfile)
    