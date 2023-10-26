import torch

class Noise():
    def __init__(self):
        pass

    def __call__(self, spec):
        mean = 0.0  # Среднее значение шума
        stddev = 0.1  # Стандартное отклонение шума

        # Генерируйте случайный шум с теми же размерами, что и ваш тензор
        noise = torch.normal(mean, stddev, spec.size())

        # Добавьте шум к тензору
        noisy_spec = spec + noise
        return noisy_spec
