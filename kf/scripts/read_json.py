import json
import numpy as np

class DataStep:
    def __init__(self, step, true_2g, noisy_true_2g, meas_2g, 
                 true_5g, noisy_true_5g, meas_5g, 
                 true_8g, noisy_true_8g, meas_8g):
        self.step = step
        self.true_2g = np.array(true_2g)
        self.noisy_true_2g = np.array(noisy_true_2g)
        self.meas_2g = np.array(meas_2g)
        self.true_5g = np.array(true_5g)
        self.noisy_true_5g = np.array(noisy_true_5g)
        self.meas_5g = np.array(meas_5g)
        self.true_8g = np.array(true_8g)
        self.noisy_true_8g = np.array(noisy_true_8g)
        self.meas_8g = np.array(meas_8g)

def parse_data(data_dict):

    data_steps = []

    for item in data_dict["data"]:
        step = item["step"]

        true_2g = item["true_2g"]
        noisy_true_2g = item["noisy_true_2g"]
        meas_2g = item["meas_2g"]

        true_5g = item["true_5g"]
        noisy_true_5g = item["noisy_true_5g"]
        meas_5g = item["meas_5g"]

        true_8g = item["true_8g"]
        noisy_true_8g = item["noisy_true_8g"]
        meas_8g = item["meas_8g"]

        data_step = DataStep(
            step=step,
            true_2g=true_2g,
            noisy_true_2g=noisy_true_2g,
            meas_2g=meas_2g,
            true_5g=true_5g,
            noisy_true_5g=noisy_true_5g,
            meas_5g=meas_5g,
            true_8g=true_8g,
            noisy_true_8g=noisy_true_8g,
            meas_8g=meas_8g,
        )

        data_steps.append(data_step)

    return data_steps

#Функция для чтения JSON из файла
def read_json(file_path: str):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def read_file(file_path: str):
    json_data = read_json(file_path)
    data = parse_data(json_data)
    return data


data = read_file("data 24-18-19 17_08_1724077092.json")
print(data)