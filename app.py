from flask import Flask, render_template, request
from sqlalchemy import create_engine
import pandas as pd
from urllib.parse import quote
import joblib

# ---------------------------------
# CREATE FLASK APP FIRST (CRITICAL)
# ---------------------------------
app = Flask(__name__)

# ---------------------------------
# LOAD MODELS
# ---------------------------------
pca_model = joblib.load("insurance_pca_model.joblib")
svd_model = joblib.load("insurance_svd_model.joblib")

# ---------------------------------
# HOME ROUTE
# ---------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# ---------------------------------
# SUCCESS ROUTE
# ---------------------------------
@app.route('/success', methods=['POST'])
def success():

    file = request.files['file']
    model_type = request.form['model']
    user = request.form['user']
    pw = quote(request.form['pw'])
    db = request.form['db']

    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

    try:
        data = pd.read_csv(file)
    except:
        data = pd.read_excel(file)

    data1 = data.drop(['Customer'], axis=1)
    num_cols = data1.select_dtypes(exclude=['object']).columns

    if model_type == 'PCA':

        pca_out = pca_model.transform(data1[num_cols])

        transformed = pd.DataFrame(
            pca_out,
            columns=[f"PC{i+1}" for i in range(pca_out.shape[1])]
        )

        final = pd.concat([data[['Customer']], transformed], axis=1)
        final.to_sql('insurance_pca_output', engine, if_exists='replace', index=False)
        title = "PCA Dimensionality Reduction Output"

    else:

        svd_out = svd_model.transform(data1[num_cols])

        transformed = pd.DataFrame(
            svd_out,
            columns=[f"SVD{i+1}" for i in range(svd_out.shape[1])]
        )

        final = pd.concat([data[['Customer']], transformed], axis=1)
        final.to_sql('insurance_svd_output', engine, if_exists='replace', index=False)
        title = "SVD Dimensionality Reduction Output"

    return render_template(
        'data.html',
        tables=final.to_html(classes='table table-striped', index=False),
        title=title
    )

# ---------------------------------
# RUN APP
# ---------------------------------
if __name__ == '__main__':
    app.run(debug=True)
