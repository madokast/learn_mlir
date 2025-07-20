import onnxruntime as ort
import numpy as np

def infer(path:str, input_data:np.ndarray): 
    # 加载 ONNX 模型
    session = ort.InferenceSession(path)

    # 获取模型输入的名称
    input_name = session.get_inputs()[0].name

    # 进行推理
    output = session.run(None, {input_name: input_data})

    return output
