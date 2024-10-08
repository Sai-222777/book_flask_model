from flask import Flask,request,jsonify
import numpy as np
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as f: 
    model = pickle.load(f)

with open('book_pivot.pkl', 'rb') as f: 
    book_pivot = pickle.load(f)


@app.route('/api/recommend', methods=['POST'])
def recommend_books():
    
    data = request.get_json()
    
    name = data['name']


    book_list = []

    # Find the book ID and calculate recommendations
    book_id = np.where(book_pivot.index == name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    
    for i in suggestion:
        for j in i:
            book_list.append(book_pivot.index[j])
    
    
    return jsonify(book_list)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)