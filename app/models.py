import tensorflow as tf
import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

class DepressionModelManager:
    def __init__(self, base_model_path, models_dir):
        self.base_model_path = base_model_path
        self.models_dir = models_dir

        #Caching models such that we do not need to reload it
        self.models = {}

    def get_model(self, user_id):
        if user_id not in self.models:
            user_model_path = os.path.join(self.models_dir, f"{user_id}.h5")
            if os.path.exists(user_model_path):
                model = tf.keras.models.load_model(user_model_path)
            else:
                model = tf.keras.models.load_model(self.base_model_path)
                model.save(user_model_path)
            self.models[user_id] = model
        return self.models[user_id]

    def evaluate_model(self, model, data, labels):
        predictions = model.predict(data)
        predictions = (predictions > 0.5).astype(int)
        f1 = f1_score(labels, predictions)
        accuracy = accuracy_score(labels, predictions)
        return accuracy, f1
    
    def vs_update_model(self, user_id, X_train, y_train, X_test, y_test):
        user_model = self.get_model(user_id)

        for layer in user_model.layers[:-1]:
            layer.trainable = False
        user_model.layers[-1].trainable = True

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        user_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        user_model.fit(X_train, y_train, epochs=500, batch_size=8) # Reduced the epochs for less intensive fine-tuning
        fine_tuned_accuracy, fine_tuned_f1 = self.evaluate_model(user_model, X_test, y_test)

        predictions_after = user_model.predict(X_test)
        predictions_after = (predictions_after > 0.5).astype(int)
        conf_matrix = confusion_matrix(y_test, predictions_after)
        report = classification_report(y_test, predictions_after)

        user_model.save(os.path.join(self.models_dir, f"{user_id}.h5"))
        return fine_tuned_accuracy, fine_tuned_f1, conf_matrix, report
    
    def update_model(self, user_id, data, labels):
        base_model = tf.keras.models.load_model(self.base_model_path)

        user_model = self.get_model(user_id)
        data = data.drop("docId", axis = 1)

        print(data.shape)
        print(data)

        data = np.array(data)
        labels = np.array(labels)

        #Split data into training and testing:
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42, stratify=labels)
        base_accuracy, base_f1 = self.evaluate_model(base_model, X_test, y_test)

        # X_train = X_train.astype('float32')
        # X_train = np.array(X_train)
        # y_train = np.array(y_train)
        # X_train = tf.convert_to_tensor(X_train)
        print("TRAINNNNNNNNN")
        print(X_train.shape)
        print(user_id)

        for layer in user_model.layers[:-1]:
            layer.trainable = False
        user_model.layers[-1].trainable = True

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        user_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        # Fine-tune user model
        user_model.fit(X_train, y_train, epochs=100, batch_size=8) # Reduced the epochs for less intensive fine-tuning
        fine_tuned_accuracy, fine_tuned_f1 = self.evaluate_model(user_model, X_test, y_test)

        predictions_after = user_model.predict(X_test)
        predictions_after = (predictions_after > 0.5).astype(int)
        conf_matrix = confusion_matrix(y_test, predictions_after)
        report = classification_report(y_test, predictions_after)
        print(conf_matrix)
        print(report)

        user_model.save(os.path.join(self.models_dir, f"{user_id}.h5"))
        return fine_tuned_accuracy, fine_tuned_f1, base_accuracy, base_f1
        # return fine_tuned_accuracy, fine_tuned_f1, base_accuracy, base_f1
    
    def get_best_model(self, user_id, model):
        if model == 'base':
            return tf.keras.models.load_model(self.base_model_path)
        else :
            return self.get_model(user_id)
    
    def preprocess(self, data):
        data_df = pd.json_normalize(data)
        print(data_df)
        data_df = data_df.drop(["userId", "timestamp.nanoseconds", "timestamp.seconds", "trained"], axis = 1)
        label_encoders = {}
        for column in data_df.columns:
            if data_df[column].dtype == 'object':
                le = LabelEncoder()
                data_df[column] = le.fit_transform(data_df[column])
                label_encoders[column] = le

        # Normalize data
        scaler = MinMaxScaler()
        data_normalized = pd.DataFrame(scaler.fit_transform(data_df), columns=data_df.columns)

        return data_normalized

    def predict(self,user_id, data, model):
        retrieved_model = self.get_best_model(user_id, model)
        return retrieved_model.predict(data)

