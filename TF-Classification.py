from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import tensorflow as tf

csv_column_names = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']
species = ['Setosa','Versicolor','Virginica']

train_path = tf.keras.utils.get_file(
    "iris_training.csv",
    "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")

test_path = tf.keras.utils.get_file(
    "iris_test.csv",
    "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=csv_column_names, header=0)
test = pd.read_csv(test_path, names=csv_column_names, header=0)

train_y = train.pop('Species')
test_y = test.pop('Species')

train.head()

def input_func(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features),labels))
    if training:
        dataset.shuffle(1000).repeat()
        
    return dataset.batch(batch_size)

my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    #two hidden layers of 30 and 10 nodes each
    hidden_units=[30,10],
    #num of classes for classification
    n_classes=3
)        
    
classifier.train(
    input_fn=lambda: input_func(train,train_y,training=True),
    #used lambda to avoid creating inner input func
    steps = 5000
)

eval_res = classifier.evaluate(
    input_fn = lambda: input_func(test,test_y,training=False)
)

print(eval_res)

def inp_function (features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

feature_names = ['SepalLength','SepalWidth','PetalLength','PetalWidth']
def make_predictions(classifier, feature_names, SPECIES):
    predict = {}
    print("Kindly Type Values as prompted: ")
    for feature in feature_names:
        valid = True
        while valid:
            val = input(feature + ": ")
            if val.replace(".", "", 1).isdigit():  
                valid = False
        predict[feature] = [float(val)]

    predictions = classifier.predict(input_fn=lambda: inp_function(predict))

    for prediction in predictions:
        class_id = prediction['class_ids'][0]
        probability = prediction['probabilities'][class_id]

        print('Prediction is "{}" ({:.1f}%)'.format(
            SPECIES[class_id], 100 * probability
        ))

make_predictions(classifier, feature_names, species)
