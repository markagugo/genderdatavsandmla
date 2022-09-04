import streamlit as st
from PIL import Image
import numpy as np 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Mac Tech Loop')
st.write("""
    ###### Exploring The Gender Dataset
    """)

image = Image.open('MCL.jpg')
st.image(image,use_column_width=True)

def main():
	activities=['EDA','Visualisation','Model']
	option =st.sidebar.selectbox('Selection option:',activities)

	if option == 'EDA':
		st.subheader("Exploratory Data Analysis")
		data = pd.read_csv('gcd.csv')
		st.dataframe(data)

		if st.checkbox("Display shape"):
				st.write(data.shape)
		if st.checkbox("Display columns"):
			st.write(data.columns)
		if st.checkbox("Select multiple columns"):
			selected_columns=st.multiselect('Select preferred columns:',data.columns)
			df1=data[selected_columns]
			st.dataframe(df1)
		if st.checkbox("Display summary"):
			st.write(data.describe().T)
		if st.checkbox('Display Correlation of data variuos columns'):
			st.write(data.corr())

	elif option=='Visualisation':
		st.subheader("Data Visualisation")
		data = pd.read_csv('gcd.csv')
		st.dataframe(data)

		if st.checkbox('Select Multiple columns to plot'):
				selected_columns=st.multiselect('Select your preferred columns',data.columns)
				df1=data[selected_columns]
				st.dataframe(df1)

		if st.checkbox('Display Heatmap'):
			st.write(sns.heatmap(df1.corr(),vmax=1,square=True,annot=True,cmap='viridis'))
			st.pyplot()
		if st.checkbox('Display Pairplot'):
			st.write(sns.pairplot(df1,diag_kind='kde'))
			st.pyplot()
		if st.checkbox('Display Pie Chart'):
			all_columns=data.columns.to_list()
			pie_columns=st.selectbox("select column to display",all_columns)
			pieChart=data[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
			st.write(pieChart)
			st.pyplot()


	elif option == 'Model':
		st.subheader("Model Building")
		data = pd.read_csv('gcd.csv')
		st.dataframe(data)

		X=data.iloc[:,0:-1]
		y=data.iloc[:,-1]

		st.sidebar.write(""" 
		#### K-NEAREST NEIGBOUR ALGORITHM
		""")

		seed=st.sidebar.slider('Seed',1,200)
		
		K=st.sidebar.slider('K',1,50)

		clf=KNeighborsClassifier(n_neighbors=K)

		X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=seed)

		clf.fit(X_train,y_train)

		y_pred=clf.predict(X_test)

		accuracy=accuracy_score(y_test,y_pred)
		
		st.write('Accuracy',accuracy)


if __name__ == '__main__':
	main() 
