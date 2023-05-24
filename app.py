from IPython.display import HTML
import folium
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim
import plotly.graph_objs as go
import plotly.offline as py
import pandas as pd
import numpy as np
import math
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flask import Flask, render_template, request, redirect, url_for
import pickle
import io
import base64
import matplotlib.pyplot as plt
plt.style.use('ggplot')

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        myData = request.form
        if myData['login'] == '1':
            return redirect(url_for('recommendation'))
        elif myData['login'] == '2':
            return redirect(url_for('consultancy'))
        else:
            return redirect(url_for('about'))
    return render_template('login.html')


@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    if request.method == 'POST':
        myData = request.form
        parameter = myData['type']
        if myData['type'] == parameter:
            fig = plt.figure(figsize=(16, 10))
            chains = df[parameter].value_counts()[:20]
            sns.barplot(x=chains, y=chains.index, palette='deep')
            plt.xlabel("Number of outlets")
            # Convert plot to PNG image
            pngImage = io.BytesIO()
            FigureCanvas(fig).print_png(pngImage)

            # Encode PNG image to base64 string
            pngImageB64String = "data:image/png;base64,"
            pngImageB64String += base64.b64encode(
                pngImage.getvalue()).decode('utf8')

            return render_template('result.html', image=pngImageB64String, parameter='Topmost Recommendations : ' + parameter, check=True)

    return render_template('recommendation.html')


@app.route('/restType', methods=['GET', 'POST'])
def restType():
    if request.method == 'POST':
        myData = request.form
        parameter = myData['restType']
        dfrt = df.groupby(['rest_type', 'name']).agg('count')
        datas = dfrt.sort_values(['url'], ascending=False).groupby(['rest_type'], as_index=False).apply(
            lambda x: x.sort_values(by="url", ascending=False).head(5))['url'].reset_index().rename(columns={'url': 'count'})

        if myData['restType'] == parameter:
            outputData = datas[datas['rest_type'] == parameter]
            outputData = outputData[['name']]
            outputData.columns = ['Restaurant Name']
            outputData = HTML(outputData.to_html(classes='table'))

            return render_template('result.html', data=outputData, rT=True, parameter=parameter)

    return render_template('recommendation.html')


@app.route('/onCuisine', methods=['GET', 'POST'])
def onCuisine():
    if request.method == 'POST':
        myData = request.form
        parameter = myData['onCuisine']
        data = pd.DataFrame(df[df['cuisines'] == parameter].groupby(
            ['name'], as_index=False)['url'].agg('count'))
        data.columns = ['Restaurants', 'Outlets']
        data = data.head()
        data = HTML(data.to_html(classes='table'))

        return render_template('result.html', data=data, oC=True, parameter=parameter)
    return render_template('recommendation.html')


@app.route('/inBudget', methods=['GET', 'POST'])
def inBudget():
    if request.method == 'POST':
        myData = request.form
        locationParameter = myData['locationf']
        rest_Type = myData['type']
        budgetRange = myData['range']
        if budgetRange == '':
            budgetRange = 1400

        def return_budget(location, rest):
            budget = cost_dist[(cost_dist['approx_cost(for two people)'] <= int(budgetRange)) & (cost_dist['location'] == location) &
                               (cost_dist['rate'] > 3) & (cost_dist['rest_type'] == rest)]
            return (budget['name'].unique())

        budgetR = return_budget(locationParameter, rest_Type)
        budgetR = budgetR[:5]
        print(type(budgetR))
        return render_template('result.html', budget=budgetR, iB=True, locationParameter=locationParameter, rest_Type=rest_Type, budgetRange=budgetRange)
    return render_template('recommendation.html')


@app.route('/consultancy', methods=['GET', 'POST'])
def consultancy():
    if request.method == 'POST':
        myData = request.form
        parameter = myData['type']

        if parameter == 'Location Vs Cuisines':
            Location = myData['Location']
            Cuisine = myData['Cuisine']
            print("df", df)
            df_1 = df.groupby(['location', 'cuisines']).agg('count')
            data = df_1.sort_values(['url'], ascending=False).groupby(['location'], as_index=False).apply(
                lambda x: x.sort_values(by="url", ascending=False).head(100))['url'].reset_index().rename(columns={'url': 'count'})
            if Location == Cuisine == 'All':
                data = data
            elif Location == 'All':
                data = data[(data['cuisines'] == Cuisine)]
            elif Cuisine == 'All':
                data = data[(data['location'] == Location)]
            else:
                data = data[(data['location'] == Location) &
                            (data['cuisines'] == Cuisine)]

            # data = data.head()
            data.columns = ['Code', 'Locations', 'Cuisines', 'Outlets']
            data = HTML(data.to_html(classes='table'))
            return render_template('result.html', data=data, parameter=parameter, clA=True)

        elif parameter == 'Cost Vs Rating':
            fig = plt.figure(figsize=(10, 7))
            sns.scatterplot(
                x="rate", y='approx_cost(for two people)', data=cost_dist)
            plt.ylabel("Cost for two people")
            plt.xlabel("Ratings")
            # plt.xticks(np.arange(0, 51, 5))
            # plt.yticks(np.arange(0, 10000, 1000))
            # Convert plot to PNG image
            pngImage = io.BytesIO()
            FigureCanvas(fig).print_png(pngImage)

            # Encode PNG image to base64 string
            pngImageB64String = "data:image/png;base64,"
            pngImageB64String += base64.b64encode(
                pngImage.getvalue()).decode('utf8')

            return render_template('result.html', image=pngImageB64String, parameter='Analysis for ' + parameter, check=True)

    return render_template('consultancy.html')


@app.route('/restaurantDensity', methods=['GET', 'POST'])
def restaurantDensity():
    if request.method == 'POST':
        myData = request.form
        parameter = myData['type']
        Location = myData['Location']
        if parameter == 'Restaurant Vs Location':
            if Location == 'All':
                data = df.loc[1:500, :]
            else:
                data = df[(df['location'] == Location)]
            locations = pd.DataFrame({"Name": data['location'].unique()})

            locations['Name'] = locations['Name'].apply(
                lambda x: "Pune " + str(x))
            lat_lon = []
            geolocator = Nominatim(user_agent="app")

            for location in locations['Name']:
                location = geolocator.geocode(location)
                if location is None:
                    lat_lon.append(np.nan)
                else:
                    geo = (location.latitude, location.longitude)
                    lat_lon.append(geo)

            locations['geo_loc'] = lat_lon
            locations.to_csv('locations.csv', index=False)
            locations["Name"] = locations['Name'].apply(
                lambda x:  x.replace("Pune", "")[1:])

            Rest_locations = pd.DataFrame(
                df['location'].value_counts().reset_index())
            Rest_locations.columns = ['Name', 'count']
            Rest_locations = Rest_locations.merge(
                locations, on='Name', how="left").dropna()
            Rest_locations['count'].max()

            def generateBaseMap(default_location=[18.52, 73.85], default_zoom_start=12):
                base_map = folium.Map(
                    location=default_location, control_scale=True, zoom_start=default_zoom_start)
                return base_map
            lat, lon = zip(*np.array(Rest_locations['geo_loc']))
            Rest_locations['lat'] = lat
            Rest_locations['lon'] = lon
            basemap = generateBaseMap()
            HeatMap(Rest_locations[['lat', 'lon', 'count']].values.tolist(
            ), zoom=20, radius=15).add_to(basemap)
            basemap.save('templates/heatmap.html')
            HTML('<iframe src=plot_data.html width=1000 height=450></iframe>')

            return render_template('heatmap.html')

    return render_template('consultancy.html')


@app.route('/about', methods=['GET', 'POST'])
def about():
    if request.method == 'POST':
        myData = request.form
        parameter = myData['type']

        if parameter == 'dataset':
            data = HTML(df.to_html(classes='table'))
            return render_template('result.html', parameter=parameter, data=data, aN=True)

    return render_template('about.html')


df = pd.read_csv('./input/pune.csv')
# df = df.loc[1:1600, :]
cost_index = df['approx_cost(for two people)'] != 'Not Present'
cost_dist = df[cost_index]
cost_dist = cost_dist[[
    'rate', 'approx_cost(for two people)', 'location', 'name', 'rest_type']]
cost_dist['rate'] = cost_dist['rate'].apply(
    lambda x: float(x if (x != '-') else np.nan))

# cost_dist=df[['rate','approx_cost(for two people)','location','name','rest_type']].dropna()
# cost_dist['rate']=cost_dist['rate'].apply(lambda x: float(x.split('/')[0]) if len(x)>3 else 0)
cost_dist['approx_cost(for two people)'] = cost_dist['approx_cost(for two people)'].apply(
    lambda x: (x.replace(',', '')))
cost_dist['approx_cost(for two people)'] = cost_dist['approx_cost(for two people)'].apply(
    lambda x: int(x.replace('₹', '')))
app.run(debug=True)
