import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import re
import requests



df = pd.read_csv('artifacts.csv')
start_year = int(df['from_num'].min())
end_year = int(df['to_num'].max())

df['type of monument'] = df['type of monument'].replace('Unknown', np.nan)
df['type of inscription'] = df['type of inscription'].replace('Unknown', np.nan)
df['material'] = df['material'].replace('Unknown', np.nan)


def clean_latin_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
df['transcription'] = df['transcription'].apply(clean_latin_text)

def bc_ad(year):
    return f'{abs(year)}BC' if year < 0 else f'{int(year)}AD'

st.set_page_config(layout='wide')

page = st.sidebar.radio("Select Page", ["Map", "Visualizations", "Latin-English"])

####### Map Page
if page == "Map":
    st.title('Ancients Inscriptions Explorer')
    st.markdown(
        'This dashboard visualizes ancient Latin inscriptions from different regions, materials, and time periods')

    st.sidebar.title('Menu')
    countries = ['All Countries'] + sorted(df['country'].dropna().unique())
    selected_country = st.sidebar.selectbox('Select Country', countries)

    df["region_size"] = df["region"].map(df["region"].value_counts())
    regions = ['All regions'] + sorted(
        df[df['country'] == selected_country]['region'].dropna().unique()) \
                                            if selected_country != 'All Countries' \
                                                else ['All regions'] + sorted(df['region'].dropna().unique())
    selected_region = st.sidebar.selectbox('Select Region', regions)

    selected_period = st.sidebar.slider('Select Time Period Range (BC to AD)', min_value=start_year, max_value=end_year,
                                        value=(start_year, end_year))

    filtered_df = df[(df['from_num'] >= selected_period[0]) & (df['to_num'] <= selected_period[1])]

    if selected_country != 'All Countries':
        filtered_df = filtered_df[filtered_df['country'] == selected_country]
    if selected_region != 'All regions':
        filtered_df = filtered_df[filtered_df['region'] == selected_region]

    fig = px.scatter_map(filtered_df, lat="lat", lon='long', color='country',
                                 color_continuous_scale=px.colors.cyclical.IceFire, zoom=4, size_max=35, width=5000,
                                 height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.text(f"""
        Summary:-
        * Total Inscriptions                       - {len(df)}
        * No. of Countries where inscription found - {len(df['country'].unique())}
        * Time range covered - From  {bc_ad(start_year)}  to   {bc_ad(end_year)}
    """)
############# map page end

############ visualizations
elif page == "Visualizations":
    st.header("Visualizations of Ancient Inscriptions")
    (column1, column2) = st.columns(2)

    with column1:
        d1 = df['country'].value_counts().reset_index()
        d1.columns = ['Country', 'No. of Inscriptions']

        fig = px.bar(d1, x='Country', y='No. of Inscriptions', title="Number of Inscriptions per Country")
        st.plotly_chart(fig)

    with column2:
        d1 = df['region'].value_counts().reset_index()
        d1.columns = ['Regions', 'No. of Inscriptions']

        fig = px.bar(d1, x='Regions', y='No. of Inscriptions', title="Number of Inscriptions per Region")
        st.plotly_chart(fig)

    (column3, column4) = st.columns(2)
    with column3:
        d2 = df['type of monument'].dropna().value_counts().head(10).reset_index()
        d2.columns = ['Monuments', 'No. of Inscriptions']
        fig = px.pie(d2, names='Monuments', values='No. of Inscriptions', title='Top 10 Most Common Monument Types')
        st.plotly_chart(fig)
    with column4:
        d2 = df['type of inscription'].dropna().value_counts().head(10).reset_index()
        d2.columns = ['name', 'No. of Inscriptions']
        fig = px.pie(d2, names='name', values='No. of Inscriptions', title='Type of inscriptions')
        st.plotly_chart(fig)

    d2 = df['material'].dropna().value_counts().head(10).reset_index()
    d2.columns = ['name', 'No. of Inscriptions']
    fig = px.pie(d2, names='name', values='No. of Inscriptions', title='Material type')
    st.plotly_chart(fig)

    material_trends = df.groupby(['from_num', 'material']).size().reset_index(name='count')
    fig = px.histogram(material_trends, x='from_num', y='count', color='material',
                       title='Trends of Inscription Materials Over Time',
                       labels={'from_num': 'Year', 'count': 'Number of Inscriptions'})
    st.plotly_chart(fig)

###########nlp
if "translate_count" not in st.session_state:
    st.session_state.translate_count = 0

MAX_TRANSLATIONS = 5

st.title("Translate the Inscriptions")
st.sidebar.title("Choose the Transcriptions")

countries = ["All Countries"] + sorted(df['country'].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Select the Country", countries)

if selected_country != "All Countries":
    df_filtered = df[df["country"] == selected_country]
    regions = ["All Regions"] + sorted(df_filtered['region'].dropna().unique().tolist())
else:
    df_filtered = df
    regions = ["All Regions"] + sorted(df["region"].dropna().unique().tolist())

selected_region = st.sidebar.selectbox("Select the Region", regions)

if selected_region != "All Regions":
    df_filtered = df_filtered[df_filtered["region"] == selected_region]

st.subheader("Latin Transcriptions:")
if df_filtered.empty:
    st.warning("No inscriptions found for the selected filters.")
    latin_texts = []
else:
    latin_texts = df_filtered["transcription"].dropna().sample(min(5, len(df_filtered))).tolist()

for idx, text in enumerate(latin_texts, start=1):
    st.text(f"{idx}. {text}")

if st.button("Translate"):
    if st.session_state.translate_count >= MAX_TRANSLATIONS:
        st.warning("Translation limit reached! You can only translate 5 times per session.")
    else:
        API_KEY = "AIzaSyBnCn9Z2cOGouQYLqKSQlG3zr7uLnZxZ-U"

        url = f"https://translation.googleapis.com/language/translate/v2?key={API_KEY}"


        payload = {
            "q": latin_texts,
            "target": "en",
            "format": "text",
            "source": "la"
        }

        response = requests.post(url, json=payload)

        if response.status_code == 200:
            result = response.json()
            if "data" in result and "translations" in result["data"]:
                translations = [t["translatedText"] for t in result["data"]["translations"]]

                st.subheader("Translated Texts:")
                for orig, trans in zip(latin_texts, translations):
                    st.write(f"**Latin:** {orig}")
                    st.write(f"**English:** {trans}")
                    st.write("---")
                st.session_state.translate_count += 1
            else:
                st.error("Error: No translation found in the response.")
        else:
            st.error(f"Request failed with status code {response.status_code}")
            st.text(response.text)

    








