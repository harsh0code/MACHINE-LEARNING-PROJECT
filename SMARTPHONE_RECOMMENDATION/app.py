from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

app = Flask(__name__)

CSV_PATH = "smartphone.csv"  # place the CSV in the same folder as app.py or adjust this path

def load_and_prepare(path=CSV_PATH):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Create a binary target "is_premium" using the 75th percentile of price
    price_q75 = df['price'].quantile(0.75)
    df['is_premium'] = (df['price'] > price_q75).astype(int)
    return df

def train_model(df):
    # select features that are numeric or easy to encode
    features = ['price','ram_capacity','internal_memory','battery_capacity','processor_speed','refresh_rate','primary_camera_rear','brand_name','processor_brand']
    df_model = df[features + ['is_premium']].copy()
    # simple impute for numeric
    num_cols = ['price','ram_capacity','internal_memory','battery_capacity','processor_speed','refresh_rate','primary_camera_rear']
    num_imputer = SimpleImputer(strategy='median')
    X_num = num_imputer.fit_transform(df_model[num_cols])
    # one-hot encode categorical small-cardinality features
    cat_cols = ['brand_name','processor_brand']
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
 
    X_cat = ohe.fit_transform(df_model[cat_cols].fillna('unknown'))
    X = np.hstack([X_num, X_cat])
    y = df_model['is_premium'].values
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, num_imputer, ohe, num_cols, cat_cols

# prepare data + model on startup
df = load_and_prepare()
model, num_imputer, ohe, num_cols, cat_cols = train_model(df)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    brand = request.form.get('brand','').strip().lower()
    max_price = request.form.get('max_price','').strip()
    try:
        max_price = float(max_price) if max_price!='' else np.inf
    except:
        max_price = np.inf

    # filter dataset
    if brand == '':
        df_filtered = df[df['price'] <= max_price].copy()
    else:
        df_filtered = df[df['brand_name'].str.contains(brand, case=False, na=False) & (df['price'] <= max_price)].copy()

    if df_filtered.empty:
        return render_template('results.html', phones=[], brand=brand, max_price=max_price, note="GAREEEEEEEEEEEEEB")

    # prepare features for prediction
    X_num = num_imputer.transform(df_filtered[num_cols])
    X_cat = ohe.transform(df_filtered[cat_cols].fillna('unknown'))
    X = np.hstack([X_num, X_cat])
    probs = model.predict_proba(X)[:,1]  # probability of being 'premium'
    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered['premium_prob'] = (probs*100).round(1)  # percent

    # convert to dicts for template
    phones = df_filtered.to_dict(orient='records')
    return render_template('results.html', phones=phones, brand=brand, max_price=max_price, note=None)

if __name__ == '__main__':
    app.run(debug=True)