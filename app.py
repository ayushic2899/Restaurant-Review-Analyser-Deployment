from flask import Flask, render_template, request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
 
c = pickle.load(open( 'restaurant_review.pkl', 'rb'))
cv = pickle.load(open('transform.pkl','rb'))

app = Flask(__name__)
@app.route('/')
def home():
	return render_template('home.html')
@app.route('/predict', methods=['POST'])

def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv.transform(data).toarray()
    	my_prediction = c.predict(vect)
    	return render_template('first.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)