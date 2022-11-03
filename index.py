from flask import Flask, render_template, request
import pandas as pd
import recommender

app = Flask(__name__)

books = pd.read_csv('dataset/Books.csv', error_bad_lines=False).drop_duplicates(subset=['Book-Title'])

@app.route('/')
def html_table():

    query = request.args.get('title', type=str)
    
    if query:
        res = books[books['Book-Title'].str.match(query, False)== True]
        reshead = res.head(10)
        return render_template('index.html',  items=reshead)

    else:
        random = books.sample(10)
        return render_template('index.html',  items=random)


@app.route('/details')
def get_book():
    isbn = request.args.get('isbn', type=str)
    res = books.loc[books['ISBN'] == isbn]

    rec = recommender.content_based_recommender(res.values[0][1])
    if (type(rec) != str) :
        rec_books = books.loc[books['Book-Title'].isin(rec)]
    else :
        rec_books = rec

    return render_template('details.html', item=res.values[0], rec=rec_books)


app.run(port=5000)