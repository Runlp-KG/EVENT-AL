import datetime
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 构建MLP模型
class EventPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EventPredictionModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.mlp(x)

def temppral_learn():
    # 训练数据
    with open('data/train_event.json', 'r') as input_file:
        events = json.load(input_file)

    # 根据时间对事件进行排序
    sorted_events = sorted(events, key=lambda event: event["time"])

    # 分割时间和事件
    event_times = []
    event_data = []
    for event in sorted_events:
        time_str = event["time"]
        if time_str == "datetime(unknown)":
            event_times.append(None)
        else:
            time_range = time_str.split("-")
            start_time = eval(time_range[0])
            event_times.append(start_time)
        event_data.append(event)

    # 构建事件之间的邻接矩阵
    num_events = len(event_data)
    # 初始化邻接矩阵
    adj_matrix = np.zeros((num_events, num_events))

    # 根据时间信息构建邻接矩阵
    for i in range(num_events):
        for j in range(i + 1, num_events):
            if event_times[i] is not None and event_times[j] is not None:
                # 如果两个事件都有明确时间，根据时间比较确定关系
                if event_times[i] < event_times[j]:
                    adj_matrix[i][j] = 1
                else:
                    adj_matrix[j][i] = 1
            elif event_times[i] is None and event_times[j] is not None:
                # 如果事件 i 时间不确定，事件 j 时间确定，则认为事件 i 在事件 j 之前
                adj_matrix[i][j] = 1
            elif event_times[i] is not None and event_times[j] is None:
                # 如果事件 i 时间确定，事件 j 时间不确定，则认为事件 j 在事件 i 之前
                adj_matrix[j][i] = 1
            else:
                # 两个事件的时间都不确定，没有确定的关系
                pass

    input_dim = num_events + 1  # 考虑事件之间的关系和时间
    hidden_dim = 128
    output_dim = num_events

    model = EventPredictionModel(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 100

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        for i in range(num_events - 1):
            # 选择事件 i 作为当前事件，事件 i+1 作为目标事件
            current_event = i
            target_event = i + 1
            
            # 根据事件时间信息和邻接矩阵构建输入数据
            input_data = np.concatenate((event_times.reshape(-1, 1), adj_matrix), axis=1)
            input_tensor = torch.FloatTensor(input_data)
            
            # 预测下一个事件的概率分布
            predicted_probs = model(input_tensor[current_event])
            
            # 创建目标事件的概率分布
            target_probs = torch.zeros_like(predicted_probs)
            target_probs[target_event] = 1.0
            
            # 计算损失并进行反向传播
            loss = criterion(predicted_probs, target_probs)
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training finished!")


    # 从 JSON 文件加载事件数据
    with open('data/test_event.json', 'r') as input_file:
        events = json.load(input_file)

    # 处理事件数据，提取事件时间和邻接矩阵信息
    event_times = []
    adj_matrix = []

    for event in events:
        time_info = event["time"]
        if time_info == "datetime(unknown)":
            event_times.append(None)
        else:
            time_range = time_info.split("-")
            start_time = eval(time_range[0])
            event_times.append(start_time)
        
        # 构建事件之间的关系矩阵
        relation_row = [event["relation"] for event in events]
        adj_matrix.append(relation_row)

    # 转换为 NumPy 数组
    event_times = np.array(event_times)
    adj_matrix = np.array(adj_matrix)

    # 做预测
    for i in range(num_events):
        if event_times[i] is None:
            input_data = np.concatenate((event_times[i], adj_matrix[i]), axis=None)
            input_tensor = torch.FloatTensor(input_data)
            predicted_probs = model(input_tensor)
            predicted_event_idx = torch.argmax(predicted_probs)
            predicted_event = event_data[predicted_event_idx]
            print(f"Predicted next event for event {i}: {predicted_event}")
        else:
            print(f"Event {i} has a known time: {event_times[i]}")

    # 对事件进行排序
    final_sorted_events = []

    for i in range(num_events):
        if event_times[i] is None:
            input_data = np.concatenate((event_times[i], adj_matrix[i]), axis=None)
            input_tensor = torch.FloatTensor(input_data)
            predicted_probs = model(input_tensor)
            predicted_event_idx = torch.argmax(predicted_probs)
            final_sorted_events.append(event_data[predicted_event_idx])
        else:
            final_sorted_events.append(event_data[i])
    
    print("Final sorted events:")
    for event in final_sorted_events:
        print(event)
    return final_sorted_events