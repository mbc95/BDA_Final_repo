import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import keras
from sklearn.preprocessing import MinMaxScaler

# load whole dataset
df_all = pd.read_csv('data/preprocessed/cleaned_data_agg_v1.csv', delimiter=",")

# sort by date
df_all["datum"] = pd.to_datetime(df_all['datum'])
df_all = df_all.set_index(df_all['datum'])
df_all = df_all.sort_index()
df_all['datum_float'] = df_all['datum'].values.astype(float)

# create train set
df = df_all['2020-05-09 00:00:00':'2021-01-29 23:59:59']

# create test set
df_test = df_all['2021-01-30 00:00:00':]

def draw_container(canvas):        
    # data
    x = active_container['datum']
    y = active_container['height_in_cm']

    plt.ioff()
    plt.figure(figsize=(7, 6))
    fig = plt.gcf()

    # 2 segments defined according to some x properties
    segment1 = (x<'2021-01-30 00:00:00')
    segment2 = (x>='2021-01-30 00:00:00')
    plt.plot(x[segment1], y[segment1], '-b', lw=2)
    plt.plot(x[segment2], y[segment2], '-r', lw=2)

    canvas = FigureCanvasTkAgg(fig, window['Graph'].Widget)
    plot_widget = canvas.get_tk_widget()
    plot_widget.grid(row=0, column=0)

def empty_container(canvas, active_container):
    # set capazity to full
    active_container.loc[active_container.tail(1).index, "height_in_cm"] = int(140)

    # draw container
    draw_container(canvas)

    return active_container

def update_timestamp(timestamp, timeValue):
    timestamp.update(value='Current Timestamp: '+str(timeValue))

def predict(active_container, active_container_test):
    timelag=5
    
    # Prepare data
    dataset_total = pd.concat((active_container.tail(timelag), active_container_test.head(1)), axis = 0)
    test_set = dataset_total.drop(columns=["hight_delta", "height_in_cm",  "datum"]).values
    y_test_set = pd.DataFrame(dataset_total["hight_delta"].values.tolist(), columns=["hight_delta"]).values

    #generate fit for train data; we need this to inverse transform the prediction
    sc1 = MinMaxScaler(feature_range = (0, 1))
    sc1.fit_transform(pd.DataFrame(active_container["hight_delta"].values.tolist(), columns=["hight_delta"]))

    #generate fit for test data
    sc = MinMaxScaler(feature_range = (0, 1))
    test_set_scaled = sc.fit_transform(test_set)

    # generate x and y test arrays
    Y_test = []
    X_test = []
    for i in range(timelag, len(test_set)):
        X_test.append(test_set_scaled[i-timelag:i, :])
        Y_test.append(y_test_set[i-1,0])
    X_test, Y_test = np.array(X_test), np.array(Y_test)

    # Prediction
    predicted_height = regressor.predict(X_test)
    predicted_height = sc1.inverse_transform(predicted_height)
    print("Pred ="+str(predicted_height)+" and real height = "+str(y_test_set))

    # add new data to frame
    active_container = pd.concat((active_container, active_container_test.head(1)), axis = 0)
    active_container_test = active_container_test.iloc[1:]
    active_container.loc[active_container.tail(1).index, "hight_delta"] = int(predicted_height)
    active_container.loc[active_container.tail(1).index, "height_in_cm"] = int(active_container.tail(2).head(1)["height_in_cm"]) + int(active_container.tail(1)["hight_delta"])
    
    print("Subtracting Height: ", active_container.loc[active_container.tail(1).index, "hight_delta"])
    print("New Height: ", active_container.loc[active_container.tail(1).index, "height_in_cm"])

    return active_container, active_container_test


matplotlib.use("TkAgg")

sg.theme("DarkTanBlue")
# Define the window layout
sz=(10,20)
col1=[  [sg.Text('Choose Container',size=(20, 1), justification='left')],
        [sg.Combo(sorted(df["container_id"].unique().astype(int)), default_value='1',key='board', enable_events=True)]]

col2=[  [sg.Text("Container 1", size=(20, 1), justification='c', key='plotHeader')], 
        [sg.Graph((640, 480), (0, 0), (640, 480), key='Graph')],
        [sg.Text('Current Timestamp: NONE',size=(50, 1), key='Timestamp',  justification='left')]]

col3=[  [sg.Button("Next Increase (one day)")],
        [sg.Button("Next Increase x7")],
        [sg.Button("Empty Container")],
        [sg.Button("Reset")],
        [sg.Button("Quit")]]

layout = [  [sg.Column(col1, vertical_alignment='top', element_justification='left' ), 
            sg.Column(col2, element_justification='c'), 
            sg.Column(col3, element_justification='c')]]

# Create the form and show it without the plot
window = sg.Window(
    "Container Simulator",
    layout,
    location=(0, 0),
    finalize=True,
    element_justification="center",
    font="Lucida 18",
)

# Add the initial plot to the window
active_container = df[df['container_id']==int(1)]
draw_container(window["Graph"].Widget)
update_timestamp(window["Timestamp"], active_container.tail(1).index.strftime('%Y-%m-%d').values[0])
# Load initial test data
active_container_test = df_test[df_test['container_id']==int(1)]

# Load LSTM Model for default Container 1
model_path = "data/modeling/models/model_container_"
regressor = keras.models.load_model(model_path+str(1))
print("LSTM model ready!")

# Create an event loop
while True:
    event, values = window.read()
    if event == 'board':
        window['plotHeader'].Update("Container "+str(values['board']))
        active_container_test = df_test[df_test['container_id']==int(values['board'])]
        active_container = df[df['container_id']==int(values['board'])]
        update_timestamp(window["Timestamp"], active_container.tail(1).index.strftime('%Y-%m-%d').values[0])
        regressor = keras.models.load_model(model_path+str(values['board']))
        print("LSTM model ready!")
        draw_container(window["Graph"].Widget)
    if event == 'Next Increase (one day)':
        active_container, active_container_test = predict(active_container, active_container_test)
        draw_container(window["Graph"].Widget)
        update_timestamp(window["Timestamp"], active_container.tail(1).index.strftime('%Y-%m-%d').values[0])
    if event == 'Next Increase x7':
        for i in range(0, 7):
            active_container, active_container_test = predict(active_container, active_container_test)
        draw_container(window["Graph"].Widget)
        update_timestamp(window["Timestamp"], active_container.tail(1).index.strftime('%Y-%m-%d').values[0])
    if event == 'Empty Container':
        active_container = empty_container(window["Graph"].Widget, active_container)
    if event == 'Reset':
        active_container_test = df_test[df_test['container_id']==int(values['board'])]
        active_container = df[df['container_id']==int(values['board'])]
        update_timestamp(window["Timestamp"], active_container.tail(1).index.strftime('%Y-%m-%d').values[0])
        draw_container(window["Graph"].Widget)
    if event == "Quit" or event == sg.WIN_CLOSED:
        break

window.close()