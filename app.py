import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("Simple Iris Flower Prediction App")
    
st.markdown("""
    ##### By: [Ankit Gupta](https://www.linkedin.com/in/ankitgupta28/):sunglasses:
    
    The ***Iris flower data set*** or ***Fisher’s Iris data set*** is one of the most famous multivariate data set used for testing various Machine Learning Algorithms.

    **Note:** If you don't see the "User Selection" sidebar, please press the `>` icon on the top left side of your screen.
    
    """)
st.image('.\iris.png')

st.subheader("""
This app predicts the **Iris flower** type!
""")

if st.sidebar.checkbox("Seaborn Pairplot",value=True):
	import seaborn as sns
    # dat = sns.load_dataset("iris")
	fig = sns.pairplot(sns.load_dataset('iris'), hue='species') 
	st.pyplot(fig)

def user_input():
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4) 
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features 

df = user_input()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train,y_train)

prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.header(f'Credits')
st.markdown("""
    **Thank you for using my application!**
    
    The dataset is often used in data mining, classification and clustering examples and to test algorithms.[Iris Dataset](https://data.covid19india.org/).
    
    This application uses the Streamlit package library. You can learn more about me and my other projects by visiting [my Github Repo] (https://github.com/Ankitg028).    
    """)