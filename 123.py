import keras as k
from keras.src.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
import os
from tkinter import *
from tkinter import ttk, filedialog
import tkinter as tk
from tkinter import messagebox


class ProgressCallback(Callback):
    def __init__(self, progress_bar, epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.epochs * 100
        self.progress_bar['value'] = progress
        root.update_idletasks()


dataset = open('dataset.txt', 'r').read().split(', ')
dataset = [float(x) for x in dataset]

train_x = np.array([dataset[0:108]])
train_y = np.array([dataset[108:120]])

future_x = np.array([])
current_dataset = dataset.copy()  # Копия исходного датасета для модификаций

model = k.Sequential()
model.add(k.layers.LSTM(1, return_sequences=True, input_shape=(108, 1)))
model.add(k.layers.Dense(32, activation='relu'))
model.add(k.layers.Dense(12))
model.compile(optimizer='adam', loss='mse')

root = Tk()
root.title("Нейросеть")
root.geometry("480x400")  # Увеличил высоту окна для дополнительных кнопок
root.resizable(False, False)
root.iconbitmap("icon.ico")

label = Label(text="Нейросеть")
buttonTrain = tk.Button(root, text="Обучить")
progressBar = ttk.Progressbar(orient="horizontal", length=440, value=0)
buttonLoad = tk.Button(root, text="Загрузить веса")
buttonSave = tk.Button(root, text="Сохранить веса", state="disabled")
buttonLoadDataset = tk.Button(root, text="Загрузить датасет", state="disabled")
buttonPredict = tk.Button(root, text="Предсказать погоду", state="disabled")
buttonTest = tk.Button(root, text="Оценить точность", state="disabled")


def load_weight():
    if os.path.exists('model.weight.h5'):
        model.load_weights('model.weight.h5')
        buttonTrain["state"] = "disable"
        buttonLoad["state"] = "disable"
        buttonSave["state"] = "normal"
        buttonPredict["state"] = "disable"
        buttonLoadDataset["state"] = "normal"
        buttonTest["state"] = "normal"


def learn_weight():
    buttonTrain["state"] = "disable"
    buttonLoad["state"] = "disable"
    buttonSave["state"] = "disable"
    buttonPredict["state"] = "disable"
    buttonLoadDataset["state"] = "disable"
    buttonTest["state"] = "disable"

    epochs = 750
    progress_callback = ProgressCallback(progressBar, epochs)

    history = model.fit(train_x, train_y, epochs=epochs, batch_size=16, verbose=2, callbacks=[progress_callback])

    plt.semilogy(history.history['loss'])
    plt.title('Loss during training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    buttonTrain["state"] = "disable"
    buttonLoad["state"] = "disable"
    buttonSave["state"] = "normal"
    buttonPredict["state"] = "disable"
    buttonLoadDataset["state"] = "normal"
    buttonTest["state"] = "normal"


def save_weight():
    model.save('model.weight.h5')


def load_dataset():
    global future_x, current_dataset

    file_path = filedialog.askopenfilename()
    if file_path:
        current_dataset = open(file_path, 'r').read().split(', ')
        current_dataset = [float(x) for x in current_dataset]
        future_x = np.array([current_dataset[12:120]])

        buttonTrain["state"] = "disable"
        buttonLoad["state"] = "disable"
        buttonSave["state"] = "normal"
        buttonPredict["state"] = "normal"
        buttonLoadDataset["state"] = "disable"
        buttonTest["state"] = "normal"


def predict_year():
    # Создаем всплывающее окно для выбора года
    popup = Toplevel()
    popup.title("Выбор года предсказания")
    popup.geometry("300x250")  # Увеличил высоту для дополнительных кнопок
    popup.resizable(False, False)

    label = Label(popup, text="Выберите на сколько лет предсказать:")
    label.pack(pady=10)

    def predict_for_years(years):
        popup.destroy()
        predict(years)

    years_options = [1, 2, 3, 4, 5]
    buttonYear1 = Button(popup, text=f"Через 1 год", command=lambda y=1: predict_for_years(y))
    buttonYear1.pack(pady=3)

    buttonYear2 = Button(popup, text=f"Через 2 года", command=lambda y=2: predict_for_years(y))
    buttonYear2.pack(pady=3)

    buttonYear3 = Button(popup, text=f"Через 3 года", command=lambda y=3: predict_for_years(y))
    buttonYear3.pack(pady=3)

    buttonYear4 = Button(popup, text=f"Через 4 года", command=lambda y=4: predict_for_years(y))
    buttonYear4.pack(pady=3)

    buttonYear5 = Button(popup, text=f"Через 5 лет", command=lambda y=5: predict_for_years(y))
    buttonYear5.pack(pady=3)

def predict(years=1):
    global current_dataset, future_x

    if years not in range(1, 6):
        messagebox.showerror("Ошибка", "Можно предсказывать только на 1-5 лет")
        return

    temp_dataset = current_dataset.copy()
    future_y = []

    for year in range(years):
        future_x = np.array([temp_dataset[-108:]])
        future_y = model.predict(future_x, verbose=0)

        if year < years - 1:
            temp_dataset = temp_dataset[12:] + future_y.tolist()[0][0]

    future_y = future_y.tolist()[0][0]
    months = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
    colors = ['blue' if value <= 0 else 'red' for value in future_y]

    plt.bar(months, future_y, color=colors)
    plt.xlabel('Месяцы')
    plt.ylabel('Температура')
    plt.title('Температура по месяцам')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.show()

    buttonTrain["state"] = "disable"
    buttonLoad["state"] = "disable"
    buttonSave["state"] = "normal"
    buttonPredict["state"] = "normal"
    buttonLoadDataset["state"] = "disable"
    buttonTest["state"] = "normal"


def testing():
    dataset_test = open('Dataset_test.txt', 'r').read().split(', ')
    dataset_test = [float(x) for x in dataset_test]

    test_x = np.array([dataset[0:108]])
    test_y = np.array([dataset[108:120]])

    score = model.evaluate(test_x, test_y, batch_size=4)
    show_custom_popup("Оценка точности", f'Оценка точности: {round(score, 3)} MSE ({round(np.sqrt(score), 3)} RMSE)')


def show_custom_popup(title: str, text: str):
    popup = Toplevel()
    popup.title(title)
    popup.geometry("450x80")
    popup.resizable(False, False)
    popup.iconbitmap("icon.ico")

    label = Label(popup, text=text)
    label.pack(pady=10)

    button = Button(popup, text="Закрыть", command=popup.destroy)
    button.pack(pady=10)


label.pack()

buttonTrain["command"] = learn_weight
buttonTrain.pack(fill=X, padx=(20, 20), pady=10)

progressBar.pack(pady=10)

buttonLoad["command"] = load_weight
buttonLoad.pack(fill=X, padx=(20, 20), pady=10)

buttonSave["command"] = save_weight
buttonSave.pack(fill=X, padx=(20, 20), pady=10)

buttonLoadDataset["command"] = load_dataset
buttonLoadDataset.pack(fill=X, padx=(20, 20), pady=10)

buttonPredict["command"] = predict_year
buttonPredict.pack(fill=X, padx=(20, 20), pady=10)

buttonTest["command"] = testing
buttonTest.pack(fill=X, padx=(20, 20), pady=10)

root.mainloop()