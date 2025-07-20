import onnxruntime as ort
import numpy as np

def infer(path, input_data:np.): 
    # 加载 ONNX 模型
    session = ort.InferenceSession(path)

    # 获取模型输入的名称
    input_name = session.get_inputs()[0].name
    print("input_name: ", input_name)

    # 进行推理
    output = session.run(None, {input_name: input_data})

    # 输出推理结果
    # output:  [array([[0.22616413, 0.28796327, 0.21973658, 0.26613602],
    #                  [0.22183387, 0.3190139 , 0.1964965 , 0.26265574]], dtype=float32)]
    print("output: ", output)
