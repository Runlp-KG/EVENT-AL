import json
import re
import string
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(predictions, labels):
    em_list = []
    for pred, label in zip(predictions, labels):
        predset = set(pred)
        labelset = set(label)
        if predset.intersection(labelset):
            em_list.append(1)
        else:
            em_list.append(0)
        # em_list_ = []
        # for label_ in label:
        #     em_list_.append(pred.lower() == label_.lower())
        # em_list.append(any(em_list_))
    em = sum(em_list) / len(predictions)
    return em
def exact_match_multiple_predictions(predicted_answers, true_answers):
    sem_list = []
    for pred_all, label_all in zip(predicted_answers, true_answers):
        pred_all = list(map(str.lower, pred_all))
        label_all = list(map(str.lower, label_all))
        # 判断是否所有预测答案都与任意一个真实答案完全匹配
        # all_match = all(pred in label_all for pred in pred_all )
        all_match = all(label in pred_all for label in label_all)
        em = 1 if all_match else 0
        sem_list.append(em)
    sem = sum(sem_list) / len(predicted_answers)
    return sem

def compute_f1(pred, labels):
    f1_list_ = []
    for label_ in labels:
        predicted_tokens = set(pred.lower().split())
        true_tokens = set(label_.lower().split())
        # 计算交集
        common_tokens = predicted_tokens.intersection(true_tokens)

        # 计算精确率和召回率
        precision = len(common_tokens) / len(predicted_tokens) if len(predicted_tokens) > 0 else 0
        recall = len(common_tokens) / len(true_tokens) if len(true_tokens) > 0 else 0
        # 计算F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_list_.append(f1)
    f1_max = max(f1_list_)
    return f1_max

def f1_score(predictions, labels):
    f1_list = []
    for preds, label in zip(predictions, labels):
        f1_list_ = []
        for pred in preds:
            pred_f1 = compute_f1(pred, label)
            f1_list_.append(pred_f1)
        f1_preds = max(f1_list_)
        f1_list.append(f1_preds)
    f1 = sum(f1_list) / len(predictions)
    return f1

def f1_score_multiple_predictions(predicted_answers, true_answers):
    f1_list = []
    for pred_list, label_list in zip(predicted_answers, true_answers):
        pred_list = list(map(str.lower, pred_list))
        label_list = list(map(str.lower, label_list))
        predicted_tokens = set(pred_list)
        true_tokens = set(label_list)
        # 计算交集
        common_tokens = predicted_tokens.intersection(true_tokens)

        # 计算精确率和召回率
        precision = len(common_tokens) / len(predicted_tokens) if len(predicted_tokens) > 0 else 0
        recall = len(common_tokens) / len(true_tokens) if len(true_tokens) > 0 else 0
        # 计算F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_list.append(f1)

    # 计算多个预测答案和多个真实答案之间的平均 F1 Score
    avg_f1 = sum(f1_list) / len(f1_list)

    return avg_f1



label_ids =[]
predictions = []


with open('predict/hard_multi_predict.json', 'r') as file:
    predict_list2 = json.load(file)

for data in predict_list2:
    # data = json.loads(line)
    idx = data['id']
    label_ids.append(data['targets'])
    predictions.append(data['prediction_text'])

em=exact_match(predictions, label_ids)
f1=f1_score(predictions, label_ids)
# em=exact_match_multiple_predictions(predictions, label_ids)
# f1=f1_score_multiple_predictions(predictions, label_ids)
print(em)
print(f1)
