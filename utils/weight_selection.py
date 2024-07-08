import torch
import numpy as np

def uniform_element_selection(wt, s_shape):
    assert wt.dim() == len(s_shape), "Tensors have different number of dimensions"
    ws = wt.clone()
    repeat_number = []
    for dim in range(wt.dim()):
        if wt.shape[dim] < s_shape[dim]:
            re_n = int(wt.shape[dim] // s_shape[dim] + 1)
            repeat_number.append(re_n)
        else:
            repeat_number.append(1)
    wt = wt.repeat(repeat_number)
    
    for dim in range(wt.dim()):
        if wt.shape[dim] % s_shape[dim] == 0:
            step = wt.shape[dim] // s_shape[dim]
            indices = torch.arange(s_shape[dim]) * step
        else:
            indices = torch.round(torch.linspace(0, wt.shape[dim]-1, s_shape[dim]))
        ws = torch.index_select(ws, dim, indices.long())
    assert ws.shape == s_shape
    return ws

import copy

def transfer(teacher, student):
    teacher_weights = teacher
    student_weights = student.state_dict()
    
    keys = copy.deepcopy(list(teacher_weights.keys()))
    for key in keys:
        if "blocks" in key:
            teacher_weights[copy.deepcopy(key).replace("blocks", "layers")] = teacher_weights.pop(key)
    
    keys = copy.deepcopy(list(teacher_weights.keys()))
    for key in keys:      
        if "final_layer" in key:
            teacher_weights[copy.deepcopy(key).replace("final_layer", "output")] = teacher_weights.pop(key)
    
    keys = copy.deepcopy(list(teacher_weights.keys()))
    for key in keys:            
        if "t_embedder" in key:
            teacher_weights[copy.deepcopy(key).replace("t_embedder", "t_embedding")] = teacher_weights.pop(key)
    
    keys = copy.deepcopy(list(teacher_weights.keys()))
    for key in keys:
        if "x_embedder" in key:
            teacher_weights[copy.deepcopy(key).replace("x_embedder", "tok_embeddings")] = teacher_weights.pop(key)
    
    keys = copy.deepcopy(list(teacher_weights.keys()))
    for key in keys:
        if ".attn" in key:
            teacher_weights[copy.deepcopy(key).replace(".attn", ".attention")] = teacher_weights.pop(key)
    
    keys = copy.deepcopy(list(teacher_weights.keys()))
    for key in keys:
        if ".mlp.fc1" in key:
            teacher_weights[copy.deepcopy(key).replace(".mlp.fc1", ".feed_forward.w1")] = teacher_weights.pop(key)
    
    keys = copy.deepcopy(list(teacher_weights.keys()))
    for key in keys:
        if ".mlp.fc2" in key:
            teacher_weights[copy.deepcopy(key).replace(".mlp.fc2", ".feed_forward.w2")] = teacher_weights.pop(key)
    
    keys = copy.deepcopy(list(teacher_weights.keys()))
    for key in keys:
        if ".qkv" in key:
            teacher_weights[copy.deepcopy(key).replace(".qkv", ".wqkv")] = teacher_weights.pop(key)
            
    keys = copy.deepcopy(list(teacher_weights.keys()))
    for key in keys:
        if ".proj" in key:
            teacher_weights[copy.deepcopy(key).replace(".proj", ".wo")] = teacher_weights.pop(key)
    
    
    weight_selection = {}
    count = 0
    for key in student_weights.keys():
        if key in teacher_weights.keys() and teacher_weights[key].dim() == student_weights[key].dim():
            weight_selection[key] = uniform_element_selection(teacher_weights[key], student_weights[key].shape)
            count += 1
        else:
            weight_selection[key] = student_weights[key]
    print(f"Initialization Success Rate: {round(count*100/len(list(student_weights.keys())),2)}%")
    return weight_selection