import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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
from sklearn.naive_bayes import GaussianNB


pd.options.mode.chained_assignment = None


def find_time(filepath):
    """
    to extract time from the path of the image
    :param filepath: the path of the image
    :return: substring containing the time information
    """
    pattern = re.compile(r'20\d+0000')
    m = pattern.search(filepath)
    if m:
        return m.group(0)
    else:
        return None


def format_time(string):
    """
    to format time to be in the form xxxx-xx-xx xx:00
    :param string: substring containing the time information
    :return: reformatted time string
    """
    return string[:4] + '-' + string[4:6] + '-' + string[6:8] + ' ' + string[8:10] + ':00'


def read_csvs(path):
    """
    to read all csv files from a folder and concatenate them into one dataframe
    :param path: the location of the folder
    :return: one dataframe containing records from all csv files
    """
    files = glob.iglob(os.path.join(path, "*.csv"))
    weather = pd.concat((pd.read_csv(f, header=14) for f in files), ignore_index=True)
    return weather


def read_images(path):
    """
    to read all the images in a folder into one big numpy array
    :param path: the location of the folder
    :return: images: one numoy array containing all the images
             filenames: a pandas Series containing all image names
    """
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
    """
    to plot the occurrence of original weather descriptions
    :param weather_column: the Series containing weather descriptions for records
    :return: None
    """
    weather_column.value_counts().plot(kind='bar')
    plt.title('occurrence for each original weather category', fontsize='xx-large')
    plt.ylabel('occurrence', fontsize='xx-large')
    plt.savefig('weather_category_count')
    #plt.show()


def cleaning(df):
    """
    to clean the dataframe such than the unique four weather categories are Rain, Clear, Cloudy and Snow
    :param df: a dataframe containing the weather records
    :return: cleaned dataframe
    """
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
    """
    to exclude weather records recorded at night and early morning.
    :param df: a dataframe containing the weather records
    :return: filtered new dataframe
    """
    df = df[(df['Time'] != '06:00') & (df['Time'] != '07:00')]
    df = df[~((df['Month'] == 10) & (df['Time'] == '19:00'))]
    df = df[~(((df['Month'] == 11) | (df['Month'] == 12) | (df['Month'] == 1))
               & ((df['Time'] == '17:00') | (df['Time'] == '18:00') | (df['Time'] == '19:00')))]
    return df


def get_image_time(image_name):
    """
    to construct a dataframe containing formatted time information from image names
    :param image_name: a dataframe containing image names
    :return: a dataframe containing formatted time information for images
    """
    image_time = image_name.apply(find_time)
    image_time = image_time.apply(format_time)
    image_time = image_time.to_frame(name='time')
    image_time.reset_index(inplace=True)
    image_time.set_index('time', inplace=True)
    return image_time


def svc_analysis(X, y, pca_parameter):
    """
    to do a SVC analysis
    :param X: training values
    :param y: target values
    :param pca_parameter: PCA parameter
    :return: None
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = make_pipeline(
        PCA(pca_parameter),
        SVC(kernel='linear', C=0.1)
    )
    model.fit(X_train, y_train)
    print("score for SVC model:", model.score(X_test, y_test))
    print(classification_report(y_test, model.predict(X_test)))


def knn_analysis(X, y, pca_parameter):
    """
    to do a KNN analysis
    :param X: training values
    :param y: target values
    :param pca_parameter: PCA parameter
    :return: None
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = make_pipeline(
        PCA(pca_parameter),
        KNeighborsClassifier(n_neighbors=8),
    )
    model.fit(X_train, y_train)
    print("score for KNN model:", model.score(X_test, y_test))


def gaussian_analysis(X, y, pca_parameter):
    """
    to do a GaussianNB analysis
    :param X: training values
    :param y: target values
    :param pca_parameter: PCA parameter
    :return: None
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = make_pipeline(
        PCA(pca_parameter),
        GaussianNB()
    )
    model.fit(X_train, y_train)
    print("score for gaussianNB model:", model.score(X_test, y_test))


def analysis(X, y, pca_parameter):
    """
    to use different machine learning models to do the analysis
    :param X: training values
    :param y: target values
    :param pca_parameter: PCA parameter
    :return: None
    """
    svc_analysis(X, y, pca_parameter)
    # knn_analysis(X, y, pca_parameter)
    # gaussian_analysis(X, y, pca_parameter)


def main():
    csv_path = sys.argv[1]
    image_path = sys.argv[2]
    weather = read_csvs(csv_path)  # read all csv files in the folder
    weather = weather.dropna(axis=1, how='all')  # drop empty flag columns
    weather = weather.drop('Data Quality', axis=1)
    weather = exclude_night(weather)
    images, filenames = read_images(image_path)  # read all images and image names in the folder
    image_times = get_image_time(filenames)

    # join two dataframes to make the image index associated with corresponding weather record
    weather_with_image = weather.join(image_times, on='Date/Time')
    weather_with_image = weather_with_image[pd.notnull(weather_with_image['index']) & pd.notnull(weather_with_image['Weather'])]
    weather_with_image['index'] = weather_with_image['index'].astype('int', copy=False)

    new_weather = weather_with_image.filter(['Weather', 'index'])  # get a new dataframe containing only needed columns
    new_weather.reset_index(inplace=True)
    new_weather = new_weather.drop('level_0', axis=1)
    plot_occurence(new_weather['Weather'])
    new_weather = cleaning(new_weather)  # do the cleaning on weather descriptions
    print(new_weather.groupby(new_weather['Weather']).count()['index'])

    index_array = np.array(new_weather['index'])
    used_images = images[index_array]  # filter out images which do not have corresponding weather records
    X = used_images.reshape(used_images.shape[0], -1)  # flatten images
    y = new_weather.as_matrix(['Weather'])
    y = np.ravel(y)
    analysis(X, y, 1000)  # PCA parameter is 1000


if __name__ == '__main__':
    main()
