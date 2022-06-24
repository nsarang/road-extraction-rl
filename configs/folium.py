"""
Each source should contain a list with the folowing items:
(authconfig, password, referer, url, username, zmax, zmin)
"""

basemaps = {
    "Google Maps": (
       "",
       "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
       19,
       0
    ),
    "Google Satellite": (
        "",
        "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        19,
        0,
    ),
    "Google Terrain": ("", "https://mt1.google.com/vt/lyrs=t&x={x}&y={y}&z={z}", 19, 0),
    "Google Terrain Hybrid": (
        "",
        "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
        19,
        0,
    ),
    "Google Satellite Hybrid": (
        "",
        "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        19,
        0,
    ),
    "Stamen Terrain": (
        "Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL",
        "http://tile.stamen.com/terrain/{z}/{x}/{y}.png",
        20,
        0,
    ),
    "Stamen Toner": (
        "Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL",
        "http://tile.stamen.com/toner/{z}/{x}/{y}.png",
        20,
        0,
    ),
    "Stamen Toner Light": (
        "Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL",
        "http://tile.stamen.com/toner-lite/{z}/{x}/{y}.png",
        20,
        0,
    ),
    "Stamen Watercolor": (
        "Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL",
        "http://tile.stamen.com/watercolor/{z}/{x}/{y}.jpg",
        18,
        0,
    ),
    "Wikimedia Map": (
        "OpenStreetMap contributors, under ODbL",
        "https://maps.wikimedia.org/osm-intl/{z}/{x}/{y}.png",
        20,
        1,
    ),
    "Wikimedia Hike Bike Map": (
        "OpenStreetMap contributors, under ODbL",
        "http://tiles.wmflabs.org/hikebike/{z}/{x}/{y}.png",
        17,
        1,
    ),
    "Esri Boundaries Places": (
        "",
        "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        20,
        0,
    ),
    "Esri Gray (dark)": (
        "",
        "http://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        16,
        0,
    ),
    "Esri Gray (light)": (
        "",
        "http://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        16,
        0,
    ),
    "Esri National Geographic": (
        "",
        "http://services.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}",
        12,
        0,
    ),
    "Esri Ocean": (
        "",
        "https://services.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
        10,
        0,
    ),
    "Esri Satellite": (
        "",
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        17,
        0,
    ),
    "Esri Standard": (
        "",
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        17,
        0,
    ),
    "Esri Terrain": (
        "",
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}",
        13,
        0,
    ),
    "Esri Transportation": (
        "",
        "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}",
        20,
        0,
    ),
    "Esri Topo World": (
        "",
        "http://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        20,
        0,
    ),
    "OpenStreetMap Standard": (
        "OpenStreetMap contributors, CC-BY-SA",
        "http://tile.openstreetmap.org/{z}/{x}/{y}.png",
        19,
        0,
    ),
    "OpenStreetMap H.O.T.": (
        "OpenStreetMap contributors, CC-BY-SA",
        "http://tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        19,
        0,
    ),
    "OpenStreetMap Monochrome": (
        "OpenStreetMap contributors, CC-BY-SA",
        "http://tiles.wmflabs.org/bw-mapnik/{z}/{x}/{y}.png",
        19,
        0,
    ),
    "Strava All": (
        "OpenStreetMap contributors, CC-BY-SA",
        "https://heatmap-external-b.strava.com/tiles/all/bluered/{z}/{x}/{y}.png",
        15,
        0,
    ),
    "Strava Run": (
        "OpenStreetMap contributors, CC-BY-SA",
        "https://heatmap-external-b.strava.com/tiles/run/bluered/{z}/{x}/{y}.png?v=19",
        15,
        0,
    ),
    "Open Weather Map Temperature": (
        "Map tiles by OpenWeatherMap, under CC BY-SA 4.0",
        "http://tile.openweathermap.org/map/temp_new/{z}/{x}/{y}.png?APPID=1c3e4ef8e25596946ee1f3846b53218a",
        19,
        0,
    ),
    "Open Weather Map Clouds": (
        "Map tiles by OpenWeatherMap, under CC BY-SA 4.0",
        "http://tile.openweathermap.org/map/clouds_new/{z}/{x}/{y}.png?APPID=ef3c5137f6c31db50c4c6f1ce4e7e9dd",
        19,
        0,
    ),
    "Open Weather Map Wind Speed": (
        "Map tiles by OpenWeatherMap, under CC BY-SA 4.0",
        "http://tile.openweathermap.org/map/wind_new/{z}/{x}/{y}.png?APPID=f9d0069aa69438d52276ae25c1ee9893",
        19,
        0,
    ),
    "CartoDb Dark Matter": (
        "Map tiles by CartoDB, under CC BY 3.0. Data by OpenStreetMap, under ODbL.",
        "http://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        20,
        0,
    ),
    "CartoDb Positron": (
        "Map tiles by CartoDB, under CC BY 3.0. Data by OpenStreetMap, under ODbL.",
        "http://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        20,
        0,
    ),
    "Bing VirtualEarth": (
        "",
        "http://ecn.t3.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1",
        19,
        1,
    ),
}
