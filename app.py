from flask import Flask, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load

app = Flask(__name__)
accuracy = None
data = pd.read_csv('apple_quality.csv')

X = data.drop('Quality', axis=1)
y = data['Quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

dump(clf, 'random_forest_model.joblib')

@app.route('/model_accuracy', methods=['GET'])
def model_accuracy():
    if accuracy is not None:
        return jsonify({
            'message': f"The model's accuracy is {accuracy}"
        })
    else:
        return jsonify({
            'message': "The model's accuracy is not available."
        })
if __name__ == '__main__':
    app.run(debug=True)
