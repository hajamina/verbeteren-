#!/usr/bin/env python
# coding: utf-8
pip install seaborn

# In[132]:


#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

#%%

#%%

employee = pd.read_csv('Employee.csv', delimiter = ',')
performancerate = pd.read_csv('PerformanceRating.csv', delimiter = ',')

#combineren van de datasets 
performancerate['ReviewDate'] = pd.to_datetime(performancerate['ReviewDate'])  # Zorg dat de aanstellingsdatum in datetime-formaat is
recent_performance = performancerate.loc[performancerate.groupby('EmployeeID')['ReviewDate'].idxmax()]
combined_dataset =  pd.merge(employee, recent_performance, on='EmployeeID', how='left')

#%%


#%%

st.markdown("""
    <h1 style='text-align: center; font-size: 40px; color: #0A1172; font-family: Arial, sans-serif;'>
        HR Dashboard 📊🚀
    </h1>
""", unsafe_allow_html=True)

# Ondertitel met CSS voor centreren
st.markdown("""
    <h2 style='text-align: center; font-size: 30px; color: #0A1172; font-family: Arial, sans-serif;'>
        Medewerkerstevredenheid en attritie
    </h2>
""", unsafe_allow_html=True)
#%%

combined_dataset['ReviewDate'] = pd.to_datetime(combined_dataset['ReviewDate'], errors='coerce')

# 1. Basisinformatie van de dataset
st.write("""
    Dit dashboard omvangt informatie over de werknemerstevredenheid en attritie. We beginnen eerst met inzicht geven over de werktevredenheidsscore
    en leggen vervolgens een verband met de attritie van de werknemers. We analyseren een uitgebreide dataset en gebruiken geavanceerde machine learning-modellen 
    om te voorspellen welke werknemers het grootste risico lopen te vertrekken. Je kunt patronen en verbanden ontdekken tussen variabelen zoals werkervaring, salaris, 
    werktevredenheid en promotiefrequentie, en zien hoe deze bijdragen aan de beslissing om te blijven of te vertrekken.
    
    Voor de HR-afdeling is dit dashboard een krachtig hulpmiddel om potentiële risico’s vroegtijdig te signaleren en gericht in te grijpen. 
    Door inzicht te krijgen in de belangrijkste drijfveren achter werknemerstevredenheid en vertrekrisico's, kan HR strategieën ontwikkelen om talent te behouden, 
    de werktevredenheid te verhogen en de algehele bedrijfsperformance te verbeteren
    """)

#%%
# Tweede lijn toevoegen
st.markdown("<hr style='border: 2px solid #0A1172; margin: 20px 0;'>", unsafe_allow_html=True)

# Tevredenheidsmetrics
st.subheader('Algemene Tevredenheid van Medewerkers')

# Gemiddelde tevredenheidscijfers
col1, col2, col3, col4, col5 = st.columns(5)

avg_job_satisfaction = combined_dataset['JobSatisfaction'].mean()
avg_env_satisfaction = combined_dataset['EnvironmentSatisfaction'].mean()
avg_relation_satisfaction = combined_dataset['RelationshipSatisfaction'].mean()
avg_manager_rating = combined_dataset['ManagerRating'].mean()
avg_self_rating = combined_dataset['SelfRating'].mean()

# Metrics visualiseren
col1.metric("Job Tevredenheid", f"{avg_job_satisfaction:.2f}/5")
col2.metric("Omgeving Tevredenheid", f"{avg_env_satisfaction:.2f}/5")
col3.metric("Relatie Tevredenheid", f"{avg_relation_satisfaction:.2f}/5")
col4.metric("Leidinggevende Beoordeling", f"{avg_manager_rating:.2f}/5")
col5.metric("Zelf Beoordeling", f"{avg_self_rating:.2f}/5")

#%%

# Box plot voor job tevredenheid per functie
st.subheader("Job Tevredenheid per Functie")

fig_box = px.box(combined_dataset, x='JobRole', y='JobSatisfaction', title="Job Tevredenheid Verdeling per Functie", 
                 labels={'JobRole': 'Functie', 'JobSatisfaction': 'Job Tevredenheid'})
fig_box.update_layout(xaxis_title='Functie', yaxis_title='Job Tevredenheid')
st.plotly_chart(fig_box)


#%%
# Tweede lijn toevoegen
st.markdown("<hr style='border: 2px solid #0A1172; margin: 20px 0;'>", unsafe_allow_html=True)

# 6. Extra: Distributie van numerieke kolommen
st.subheader("Distributie van numerieke kolommen")
numeric_columns = combined_dataset.select_dtypes(include=['float64', 'int64']).columns
selected_column = st.selectbox("Kies een numerieke kolom om te visualiseren:", numeric_columns)

# Voeg een slider toe waarmee de gebruiker het aantal bins kan instellen
bins = st.slider("Selecteer het aantal bins voor het histogram", min_value=5, max_value=50, value=20)

fig, ax = plt.subplots()
# Plot het histogram met de gekozen hoeveelheid bins
sns.histplot(combined_dataset[selected_column], bins=bins, kde=True, ax=ax)
ax.set_title(f"Histogram van {selected_column} (met {bins} bins)")
st.pyplot(fig)

#%%

combined_dataset['HireDate']=combined_dataset['HireDate'].astype('datetime64[ns]')

#%%

combined_dataset.insert(0, 'FullName', combined_dataset['FirstName'] + ' ' + combined_dataset['LastName'])  # Plaats op positie 0 (eerste kolom)

# Verwijder de originele kolommen 'FirstName', 'LastName' en andere onodige kolomen 
combined_dataset.drop(columns=['FirstName', 'LastName','SelfRating','ManagerRating'], inplace=True)

#%%

# Meerdere kolommen verwijderen
combined_dataset = combined_dataset.drop(columns=['EmployeeID', 'PerformanceID'])

#%%
# Tweede lijn toevoegen
st.markdown("<hr style='border: 2px solid #0A1172; margin: 20px 0;'>", unsafe_allow_html=True)

# Bekijk de unieke waarden in de 'Gender' kolom
print(combined_dataset['Gender'].unique())
print(combined_dataset['BusinessTravel'].unique())
print(combined_dataset['Attrition'].unique())
print(combined_dataset['Department'].unique())

# Verwijder leidende en volgende spaties in 'BusinessTravel' en 'Gender'
combined_dataset['BusinessTravel'] = combined_dataset['BusinessTravel'].str.strip()
combined_dataset['Gender'] = combined_dataset['Gender'].str.strip()

# Omzetten van de volgende kolomen om in numerieke warden 
combined_dataset['Gender'] = combined_dataset['Gender'].map({"Prefer Not To Say" :0, "Male": 1, "Female": 2, "Non-Binary":3})
combined_dataset['BusinessTravel'] = combined_dataset['BusinessTravel'].map({"No Travel": 0, "Some Travel": 1, "Frequent Traveller": 2})
combined_dataset['Attrition'] = combined_dataset['Attrition'].map({'Yes': 1, 'No': 0})

#Een nieuwe variabele toevoegen aan dataset
combined_dataset['PromotionFrequency'] = (combined_dataset['YearsAtCompany'] / (combined_dataset['YearsSinceLastPromotion'] + 1)).round().astype(int)

gps_coordinates = {
    'IL': (40.6331, -89.3985),   # Coördinaten van Springfield, IL
    'CA': (36.7783, -119.4179), # Coördinaten van Sacramento, CA
    'NY': (40.7128, -74.0060)     # Coördinaten van Albany, NY
}

# Woordenboeken voor latitude (breedtegraad) en longitude (lengtegraad)
latitude_dict = {State: gps_coordinates[State][0] for State in gps_coordinates}
longitude_dict = {State: gps_coordinates[State][1] for State in gps_coordinates}

# Latitude en longitude kolommen toevoegen aan de DataFrame
combined_dataset['latitude'] = combined_dataset['State'].map(latitude_dict)
combined_dataset['longitude'] = combined_dataset['State'].map(longitude_dict)

pd.isna(combined_dataset).sum() # aantal missende waarden per kolom  
combined_dataset.shape


# In[134]:


#combined_dataset.head()


# In[148]:


#from folium.plugins import MarkerCluster
#m = folium.Map(location=[37.0902, -95.7129], zoom_start=4)  # Pas aan voor jouw dataset

# Voeg een marker cluster toe aan de kaart
#marker_cluster = MarkerCluster().add_to(m)

#for idx, row in combined_dataset.iterrows():
    #folium.CircleMarker(
      #  location=[row['latitude'], row['longitude']],
       # radius=5,
        #color='red' if row['Attrition'] == 1 else 'green',
        #fill=True,
        #fill_color='red' if row['Attrition'] == 1 else 'green',
        #fill_opacity=0.7,
        #popup=f"Afdeling: {row['Department']}"
    #).add_to(marker_cluster)

#m


# In[154]:


#pip install folium


# In[158]:


import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from collections import Counter

# Voorbeeld dataset laden (vervang dit met je eigen dataset)
# combined_dataset = pd.read_csv('your_dataset.csv')

# Definieer unieke kleuren per staat
state_colors = {
    'IL': 'blue',    # Illinois
    'CA': 'orange',  # California
    'NY': 'purple'   # New York
}

# Interactieve widgets voor filtering
selected_states = st.multiselect(
    "Selecteer Staten:",
    options=combined_dataset['State'].unique(),
    default=combined_dataset['State'].unique()
)

attrition_filter = st.radio(
    "Toon Werknemers Die:",
    ('Blijven (Attrition = 0)', 'Vertrekken (Attrition = 1)', 'Alle')
)

# Dataset filteren op basis van gebruikersselectie
filtered_data = combined_dataset[combined_dataset['State'].isin(selected_states)]

if attrition_filter == 'Blijven (Attrition = 0)':
    filtered_data = filtered_data[filtered_data['Attrition'] == 0]
elif attrition_filter == 'Vertrekken (Attrition = 1)':
    filtered_data = filtered_data[filtered_data['Attrition'] == 1]

# Folium kaart maken
m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)  # USA centrale coördinaten

# Marker Cluster toevoegen
marker_cluster = MarkerCluster().add_to(m)

# Markers toevoegen op basis van gefilterde data
for idx, row in filtered_data.iterrows():
    state_color = state_colors.get(row['State'], 'gray')  # Kleur per staat
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=7,
        color=state_color,
        fill=True,
        fill_color=state_color,
        fill_opacity=0.7,
        popup=f"Department: {row['Department']}, Years at Company: {row['YearsAtCompany']}, Attrition: {'Vertrokken' if row['Attrition'] == 1 else 'Gebleven'}"
    ).add_to(marker_cluster)

# Legenda toevoegen aan de kaart
legend_html = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 250px; height: 120px; 
     background-color: white; z-index:9999; font-size:14px;
     border:2px solid grey; border-radius:6px; padding: 10px;">
     <strong>Legenda:</strong><br>
     <i style="background:green; width: 10px; height: 10px; float:left; margin-right: 5px;"></i> Gebleven <br>
     <i style="background:red; width: 10px; height: 10px; float:left; margin-right: 5px;"></i> Vertrokken <br>
     <i style="background:blue; width: 10px; height: 10px; float:left; margin-right: 5px;"></i> Illinois (IL) <br>
     <i style="background:orange; width: 10px; height: 10px; float:left; margin-right: 5px;"></i> Californië (CA) <br>
     <i style="background:purple; width: 10px; height: 10px; float:left; margin-right: 5px;"></i> New York (NY) <br>
     </div>
     '''
m.get_root().html.add_child(folium.Element(legend_html))

# Kaart weergeven in Streamlit
st_folium(m, width=725)

# Summary-berekeningen
total_employees_in_selected_states = len(filtered_data)
leaving_employees = filtered_data[filtered_data['Attrition'] == 1]
num_leaving_employees = len(leaving_employees)

if num_leaving_employees > 0:
    avg_years_at_company_leaving = leaving_employees['YearsAtCompany'].mean()
    most_common_department = Counter(leaving_employees['Department']).most_common(1)[0][0]
else:
    avg_years_at_company_leaving = 0
    most_common_department = "N/A"

# Samenvatting in het Nederlands onder de kaart weergeven
st.write(f"**Totaal aantal werknemers in geselecteerde staten:** {total_employees_in_selected_states}")
st.write(f"**Aantal vertrekkende werknemers:** {num_leaving_employees}")
st.write(f"**Gemiddeld aantal jaren bij het bedrijf (vertrekkende werknemers):** {avg_years_at_company_leaving:.2f}")
st.write(f"**Meest voorkomende afdeling (vertrekkende werknemers):** {most_common_department}")


# In[ ]:




