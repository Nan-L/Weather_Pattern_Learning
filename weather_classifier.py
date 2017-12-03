import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sys
import matplotlib.image as mpimg
import glob
import os
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


pd.options.mode.chained_assignment = None



def find_time(filepath):
    pattern = re.compile(r'20\d+0000')
    m = pattern.search(filepath)
    if m:
        return m.group(0)
    else:
        return None


def format_time(string):
    return string[:4] + '-' + string[4:6] + '-' + string[6:8] + ' ' + string[8:10] + ':00'


# read from a folder and concatenate into a dataframe
def read_csvs(path):
    path = r'./weather'
    files = glob.iglob(os.path.join(path, "*.csv"))
    weather = pd.concat((pd.read_csv(f, header=14) for f in files), ignore_index=True)
    return weather


# read all the images in the folder into one big numpy array, return both this array and the series containin image names
def read_images(path):
    files = glob.iglob(os.path.join(path, "*.jpg"))
    collection = []
    collection_names = []
    for f in files:
        image = mpimg.imread(f)
        collection.append(image)
        collection_names.append(f)
    images = np.array(collection)
    filenames = pd.Series(collection_names)
    return images, filenames


def plot_occurence(weather_column):
    weather_column.value_counts().plot(kind='bar')
    plt.title('occurrence for each original weather category', fontsize='xx-large')
    plt.ylabel('occurrence', fontsize='xx-large')
    plt.savefig('weather_category_count')
    #plt.show()


def cleaning(df):
    df['Weather'] = df['Weather'].str.replace('Moderate ', '')
    df['Weather'] = df['Weather'].str.replace(' Showers', '')
    df['Weather'] = df['Weather'].str.replace('Mainly ', '')
    df['Weather'] = df['Weather'].str.replace('Mostly ', '')
    df = df.groupby('Weather').filter(lambda x: len(x) >= 10)
    df['Weather'] = df['Weather'].str.replace('Drizzle', 'Rain')
    df = df[df['Weather'] != 'Fog']
    df = df[df['Weather'] != 'Rain,Fog']
    return df


def exclude_night(df):
    df = df[(df['Time'] != '06:00') & (df['Time'] != '07:00')]
    df = df[~((df['Month'] == 10) & (df['Time'] == '19:00'))]
    df = df[~(((df['Month'] == 11) | (df['Month'] == 12) | (df['Month'] == 1))
               & ((df['Time'] == '17:00') | (df['Time'] == '18:00') | (df['Time'] == '19:00')))]
    return df


def ml_analysis(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = make_pipeline(
        PCA(1000),
        SVC(kernel='linear', C=0.1)
    )
    model.fit(X_train, y_train)
    print("score for SVC model:", model.score(X_test, y_test))
    print(classification_report(y_test, model.predict(X_test)))


def knn_analysis(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = make_pipeline(
        PCA(1500),
        KNeighborsClassifier(n_neighbors=5),
    )
    model.fit(X_train, y_train)
    print("score for knn model:", model.score(X_test, y_test))
    print(classification_report(y_test, model.predict(X_test)))

def main():
    csv_path = r'./weather'
    weather = read_csvs(csv_path)
    weather = weather.dropna(axis=1, how='all')  # drop useless flag columns
    weather = weather.drop('Data Quality', axis=1)

    # read all the images in the folder into one big numpy array
    image_path = r'./images'
    images, filenames = read_images(image_path)

    filenames = filenames.apply(find_time)
    filenames = filenames.apply(format_time)
    filenames = filenames.to_frame(name='time')
    filenames.reset_index(inplace=True)
    filenames.set_index('time', inplace=True)

    weather = exclude_night(weather)

    weather_with_image = weather.join(filenames, on='Date/Time')
    weather_with_image = weather_with_image[pd.notnull(weather_with_image['index']) & pd.notnull(weather_with_image['Weather'])]
    weather_with_image['index'] = weather_with_image['index'].astype('int', copy=False)

    new_weather = weather_with_image.filter(['Weather', 'index'])
    new_weather.reset_index(inplace=True)
    new_weather = new_weather.drop('level_0', axis=1)
    plot_occurence(new_weather['Weather'])

    new_weather = cleaning(new_weather)
    print(new_weather.groupby(new_weather['Weather']).count()['index'])
    print(new_weather.shape)

    index_array = np.array(new_weather['index'])
    used_images = images[index_array]
    X = used_images.reshape(used_images.shape[0], -1)
    y = new_weather.as_matrix(['Weather'])
    y = np.ravel(y)
    ml_analysis(X, y)
    #knn_analysis(X, y)



if __name__ == '__main__':
    main()
