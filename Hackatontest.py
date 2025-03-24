import requests
import pandas as pd
import numpy as np
import streamlit as st
import folium
import ast
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from folium.plugins import HeatMap
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import seaborn as sns  # (optioneel, als je geen seaborn wilt gebruiken, kun je het eruit laten)
from folium.plugins import HeatMapWithTime
from branca.element import Template, MacroElement

# === 1. TITEL VAN JE APP ===
st.title("Vluchtdata uit Sensornet API")

# === 2. DATA INLADEN & VOORBEWERKING ===
default_start = '2025-03-01'
default_end = '2025-03-08'

@st.cache_data(show_spinner="📡 Data ophalen van Sensornet API...")
def laad_sensornet_data(start_date, end_date):
    start_ts = int(pd.to_datetime(start_date).timestamp())
    end_ts = int(pd.to_datetime(end_date).timestamp())

    url = f'https://sensornet.nl/dataserver3/event/collection/nina_events/stream?conditions%5B0%5D%5B%5D=time&conditions%5B0%5D%5B%5D=%3E%3D&conditions%5B0%5D%5B%5D={start_ts}&conditions%5B1%5D%5B%5D=time&conditions%5B1%5D%5B%5D=%3C&conditions%5B1%5D%5B%5D={end_ts}&conditions%5B2%5D%5B%5D=label&conditions%5B2%5D%5B%5D=in&conditions%5B2%5D%5B2%5D%5B%5D=21&conditions%5B2%5D%5B2%5D%5B%5D=32&conditions%5B2%5D%5B2%5D%5B%5D=33&conditions%5B2%5D%5B2%5D%5B%5D=34&args%5B%5D=aalsmeer&args%5B%5D=schiphol&fields%5B%5D=time&fields%5B%5D=location_short&fields%5B%5D=location_long&fields%5B%5D=duration&fields%5B%5D=SEL&fields%5B%5D=SELd&fields%5B%5D=SELe&fields%5B%5D=SELn&fields%5B%5D=SELden&fields%5B%5D=SEL_dB&fields%5B%5D=lasmax_dB&fields%5B%5D=callsign&fields%5B%5D=type&fields%5B%5D=altitude&fields%5B%5D=distance&fields%5B%5D=winddirection&fields%5B%5D=windspeed&fields%5B%5D=label&fields%5B%5D=hex_s&fields%5B%5D=registration&fields%5B%5D=icao_type&fields%5B%5D=serial&fields%5B%5D=operator&fields%5B%5D=tags'
    response = requests.get(url)
    colnames = pd.DataFrame(response.json()['metadata'])
    data = pd.DataFrame(response.json()['rows'])
    data.columns = colnames.headers
    data['time'] = pd.to_datetime(data['time'], unit='s')
    return data

# Data laden
data = laad_sensornet_data(default_start, default_end)
data['time'] = pd.to_datetime(data['time'])

# === 3. SIDEBAR: KEUZE VAN VISUALISATIE/SECTIE ===
st.sidebar.header("Navigatie")
keuze = st.sidebar.selectbox(
    "Kies een onderdeel",
    [
        "Dataoverzicht",
        "Heatmap geluid (per uur)",
        "Geluidsvergelijking per vliegtuigtype",
        "Hoogte vs geluid (regressie)",
        "Bouwjaar vs geluid",
        "Grootteklasse vs geluid",
        "Stijgend vs dalend"
    ]
)

# === 4. GEMEENSCHAPPELIJKE FILTERS: DATUM & LOCATIES ===
# (je kunt er ook voor kiezen filters per pagina te zetten als die erg verschillend zijn)

st.sidebar.subheader("Filters")

start_date = st.sidebar.date_input("Startdatum", data['time'].min().date())
end_date = st.sidebar.date_input("Einddatum", data['time'].max().date())

# Filter de data
df = data[(data['time'].dt.date >= start_date) & (data['time'].dt.date <= end_date)]

# Locatiefilter (optioneel)
if 'location_short' in df.columns:
    locaties = df['location_short'].dropna().unique().tolist()
    geselecteerde_locaties = st.sidebar.multiselect("Locatie(s) selecteren", locaties, default=locaties)
    df = df[df['location_short'].isin(geselecteerde_locaties)]

# Voor eventuele lat/lon-parsing (nodig voor de heatmap):
def parse_location(loc):
    try:
        if isinstance(loc, str):
            return ast.literal_eval(loc)
        elif isinstance(loc, list):
            return loc
    except:
        return [None, None]

df[['lat', 'lon']] = df['location_long'].apply(parse_location).apply(pd.Series)

# === 5. PAGINA: DATAOVERZICHT ===
if keuze == "Dataoverzicht":
    st.header("🔍 Algemene informatie en data")
    st.markdown(f"Aantal rijen: **{len(df)}** | Aantal kolommen: **{df.shape[1]}**")
    st.markdown("**Kolommen:**")
    st.json(df.columns.to_list())

    st.subheader("📋 Dataoverzicht (eerste 100 rijen)")
    st.dataframe(df.head(100))

    st.write("📅 Periode in deze subset:", df['time'].min(), "tot", df['time'].max())


# === 6. PAGINA: HEATMAP GELUID (PER UUR) ===
elif keuze == "Heatmap geluid (per uur)":
    st.header("🔥 Geluidsheatmap per tijdstip")

    # Maken van folium-kaart met puntrandering o.b.v. SEL_dB
    # (code grotendeels overgenomen uit je voorbeeld)

    # Sensorlocaties die niet in 'location_long' staan, kun je evt. hardcoden:
    sensor_coords = {
        'Aa': [52.263, 4.750],
        'Bl': [52.271, 4.785],
        'Cn': [52.290, 4.725],
        'Ui': [52.245, 4.770],
        'Ho': [52.287, 4.780],
        'Da': [52.310, 4.740],
        'Ku': [52.275, 4.760],
        'Co': [52.265, 4.730],
    }

    # Zorg dat je in df de juiste lat/lon hebt:
    df['lat'] = df['location_short'].map(lambda x: sensor_coords.get(x, [None, None])[0])
    df['lon'] = df['location_short'].map(lambda x: sensor_coords.get(x, [None, None])[1])

    df['hour'] = df['time'].dt.hour
    geselecteerd_uur = st.slider("🕒 Kies een uur", min_value=0, max_value=23, value=12)

    filtered = df[(df['hour'] == geselecteerd_uur)].dropna(subset=['lat', 'lon', 'SEL_dB'])
    st.write(f"🔎 Aantal meetpunten om {geselecteerd_uur}:00 uur: {len(filtered)}")

    m = folium.Map(location=[52.3, 4.75], zoom_start=11)

    def dB_naar_kleur(sel_db, min_dB=30, max_dB=70):
        if max_dB == min_dB:
            return 'rgb(255,255,0)'
        norm = min(max((sel_db - min_dB) / (max_dB - min_dB), 0), 1)
        rood = int(255 * norm)
        groen = int(255 * (1 - norm))
        return f'rgb({rood},{groen},0)'

    min_dB = filtered['SEL_dB'].min()
    max_dB = filtered['SEL_dB'].max()

    for _, row in filtered.iterrows():
        kleur = dB_naar_kleur(row['SEL_dB'], min_dB, max_dB)
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=8,
            color=kleur,
            fill=True,
            fill_color=kleur,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"📍 {row['location_short']}<br>🔊 SEL_dB: {round(row['SEL_dB'],1)} dB",
                max_width=200
            )
        ).add_to(m)

    # Legenda
    legend_html = """
    {% macro html() %}
    <div style='position: fixed; 
         bottom: 30px; left: 30px; width: 200px; height: 130px;
         border:2px solid grey; z-index:9999; font-size:14px;
         background-color:white; padding: 10px; border-radius: 10px;'>
        <b>🎨 Kleurverloop (SEL_dB)</b><br>
        <i style="color:rgb(0,255,0);">●</i> Laag geluid (stil)<br>
        <i style="color:rgb(255,255,0);">●</i> Gemiddeld geluid<br>
        <i style="color:rgb(255,0,0);">●</i> Hoog geluid (overlast)
    </div>
    {% endmacro %}
    """
    legend = MacroElement()
    legend._template = Template(legend_html)
    m.get_root().add_child(legend)

    st_folium(m, width=750, height=500)


# === 7. PAGINA: GELUIDSVERGELIJKING PER VLIEGTUIGTYPE (capacititeiten) ===
elif keuze == "Geluidsvergelijking per vliegtuigtype":
    st.header("📊 Geluidsvergelijking per vliegtuigtype")

    # Voorbeeld zoals in jouw code
    data_clean = df.dropna(subset=["icao_type", "SEL_dB"])

    # Capaciteitstabel
    capaciteit_data = {
        "A320": {"passagiers": 180, "vracht_ton": 2.5},
        "A319": {"passagiers": 140, "vracht_ton": 2.0},
        "A321": {"passagiers": 220, "vracht_ton": 3.0},
        "B738": {"passagiers": 189, "vracht_ton": 2.6},
        "B737": {"passagiers": 162, "vracht_ton": 2.4},
        "B744": {"passagiers": 416, "vracht_ton": 20.0},
        "B77W": {"passagiers": 396, "vracht_ton": 23.0},
        "B77F": {"passagiers": 0,   "vracht_ton": 112.0},
        "A332": {"passagiers": 278, "vracht_ton": 15.0},
        "A333": {"passagiers": 277, "vracht_ton": 16.0},
        "E190": {"passagiers": 100, "vracht_ton": 1.5},
        "CRJ2": {"passagiers": 50,  "vracht_ton": 0.8},
    }
    capaciteit_df = pd.DataFrame(capaciteit_data).T.reset_index()
    capaciteit_df.columns = ["icao_type", "passagiers", "vracht_ton"]

    gemiddeld_geluid = data_clean.groupby("icao_type")["SEL_dB"].mean().reset_index()
    gemiddeld_geluid.columns = ["icao_type", "gemiddeld_SEL_dB"]

    resultaat = pd.merge(gemiddeld_geluid, capaciteit_df, on="icao_type", how="left")
    resultaat["geluid_per_passagier"] = resultaat["gemiddeld_SEL_dB"] / resultaat["passagiers"]
    resultaat["geluid_per_ton_vracht"] = resultaat["gemiddeld_SEL_dB"] / resultaat["vracht_ton"]
    resultaat_sorted = resultaat.sort_values(by="geluid_per_passagier")

    optie = st.radio("Kies wat je wilt vergelijken:", ["Per passagier", "Per ton vracht"])
    if optie == "Per passagier":
        kolom = "geluid_per_passagier"
        titel = "Geluidsbelasting per passagier"
    else:
        kolom = "geluid_per_ton_vracht"
        titel = "Geluidsbelasting per ton vracht"

    resultaat_plot = resultaat_sorted.replace([float('inf'), -float('inf')], np.nan).dropna(subset=[kolom])
    resultaat_plot = resultaat_plot.sort_values(by=kolom).reset_index(drop=True)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Kleurverloop maken (optioneel, of gebruik gewoon 1 kleur)
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    norm = colors.Normalize(
        vmin=resultaat_plot[kolom].min(),
        vmax=resultaat_plot[kolom].max()
    )
    cmap = cm.get_cmap("RdYlGn_r")  
    kleuren = [cmap(norm(x)) for x in resultaat_plot[kolom]]

    barplot = sns.barplot(
        data=resultaat_plot,
        x=kolom,
        y="icao_type",
        ax=ax,
        ci=None
    )

    # Kleuren toekennen
    for i, patch in enumerate(barplot.patches):
        patch.set_facecolor(kleuren[i])

    ax.set_xlabel("Gemiddelde geluidsbelasting (dB)")
    ax.set_ylabel("Vliegtuigtype")
    ax.set_title(titel)
    st.pyplot(fig)

    # Tabel
    st.markdown("**Onderliggende gegevens**")
    st.dataframe(
        resultaat_plot[
            [
                "icao_type",
                "gemiddeld_SEL_dB",
                "passagiers",
                "vracht_ton",
                "geluid_per_passagier",
                "geluid_per_ton_vracht"
            ]
        ]
    )


# === 8. PAGINA: HOOGTE VS GELUID (REGRESSIE) ===
elif keuze == "Hoogte vs geluid (regressie)":
    st.header("✈️ Hoe hoger het vliegtuig, hoe lager het geluid")

    df_hoogte = df.dropna(subset=["SEL_dB", "altitude"])
    if len(df_hoogte) < 2:
        st.warning("Te weinig data voor een regressieplot.")
    else:
        X = df_hoogte["altitude"].values.reshape(-1, 1)
        y = df_hoogte["SEL_dB"].values

        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_

        x0, x1 = 1000, 2000
        y0 = model.predict([[x0]])[0]
        y1 = model.predict([[x1]])[0]
        delta_y = y0 - y1

        df_hoogte["hoogte_binned"] = (df_hoogte["altitude"] // 100) * 100
        percentielen = df_hoogte.groupby("hoogte_binned")["SEL_dB"].quantile([0.1, 0.5, 0.9]).unstack()
        percentielen = percentielen.reset_index()
        percentielen.columns = ["Hoogte", "P10", "P50", "P90"]
        percentielen["Hoogte"] = percentielen["Hoogte"].astype(float)

        fig, ax = plt.subplots(figsize=(10, 6))
        # Spreidingsband
        ax.fill_between(percentielen["Hoogte"], percentielen["P10"], percentielen["P90"],
                        alpha=0.2, color='gray', label="10–90% spreiding")
        ax.plot(percentielen["Hoogte"], percentielen["P50"], color='gray', linestyle="--", label="Mediaan")

        sns.scatterplot(data=df_hoogte, x="altitude", y="SEL_dB", alpha=0.2, ax=ax, color="orange", label="Meetpunten")
        sns.regplot(data=df_hoogte, x="altitude", y="SEL_dB", scatter=False, color="red", ax=ax, label="Gem. trend")

        ax.annotate(f"{delta_y:.1f} dB verschil", xy=((x0 + x1) / 2, (y0 + y1) / 2),
                    xytext=(x0 + 150, y0 + 1.5), arrowprops=dict(arrowstyle="->", color="black"),
                    fontsize=10, color="black")

        ax.set_title("✈️ Hoe hoger het vliegtuig, hoe lager het geluid", fontsize=14)
        ax.set_xlabel("Hoogte (m)", fontsize=12)
        ax.set_ylabel("Geluidsniveau (SEL_dB)", fontsize=12)
        ax.legend()
        plt.tight_layout()

        st.pyplot(fig)

        st.markdown(f"""
        🔍 Deze grafiek toont het verband tussen **vlieghoogte** en **geluidsniveau**.
        
        - De rode lijn is de **gemiddelde trend**: geluid daalt gemiddeld **{delta_y:.1f} dB tussen 1000 en 2000 meter**.
        - De grijze band laat de spreiding per hoogte zien.
        """)


# === 9. PAGINA: BOUWJAAR VS GELUID ===
elif keuze == "Bouwjaar vs geluid":
    st.header("📈 Technologieklasse: bouwjaar vs geluid")

    # Eerst kleine merge voor bouwjaren
    bouwjaren = {
        "A320": 1988,
        "A319": 1995,
        "A321": 1993,
        "B738": 1997,
        "B737": 1968,
        "B744": 1989,
        "B77W": 2004,
        "B77F": 2008,
        "A332": 1992,
        "A333": 1993,
        "E190": 2004,
        "CRJ2": 1995,
    }
    bouwjaar_df = pd.DataFrame(list(bouwjaren.items()), columns=["icao_type", "eerste_vlucht_jaar"])
    df_uitgebreid = df.merge(bouwjaar_df, on="icao_type", how="left")

    df_tech = df_uitgebreid.dropna(subset=["eerste_vlucht_jaar", "SEL_dB", "icao_type"])
    if df_tech.empty:
        st.warning("Geen data beschikbaar om bouwjaar vs geluid te tonen.")
    else:
        gemiddeld_per_type = df_tech.groupby(["icao_type", "eerste_vlucht_jaar"])["SEL_dB"].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=gemiddeld_per_type, x="eerste_vlucht_jaar", y="SEL_dB", hue="icao_type", s=100, ax=ax)
        sns.regplot(data=gemiddeld_per_type, x="eerste_vlucht_jaar", y="SEL_dB", scatter=False, color="red", label="Trend")

        ax.set_title("Technologieklasse: Nieuwere vliegtuigen maken minder geluid", fontsize=14)
        ax.set_xlabel("Eerste vluchtjaar vliegtuigtype", fontsize=12)
        ax.set_ylabel("Gemiddeld geluidsniveau (SEL_dB)", fontsize=12)
        ax.legend()
        st.pyplot(fig)

        st.markdown("""
        ✈️ Elk punt is een vliegtuigtype met bijbehorend geluidsniveau.  
        🔴 De rode lijn toont de gemiddelde trend: hoe nieuwer het vliegtuigtype, hoe lager het geluid.
        """)


# === 10. PAGINA: GROOTTEKLASSE VS GELUID ===
elif keuze == "Grootteklasse vs geluid":
    st.header("🛩️ Grootteklasse: geluidsniveau per grootteklasse")

    # Groeperen
    grootteklasse = {
        "CRJ2": "Klein",
        "E190": "Klein",
        "A319": "Middelgroot",
        "A320": "Middelgroot",
        "A321": "Middelgroot",
        "B738": "Middelgroot",
        "A332": "Groot",
        "A333": "Groot",
        "B744": "Groot",
        "B77W": "Groot",
        "B77F": "Groot"
    }
    grootteklasse_df = pd.DataFrame(list(grootteklasse.items()), columns=["icao_type", "grootteklasse"])
    df_uitgebreid = df.merge(grootteklasse_df, on="icao_type", how="left")

    df_grootte = df_uitgebreid.dropna(subset=["grootteklasse", "SEL_dB"])

    if df_grootte.empty:
        st.warning("Geen data beschikbaar voor grootteklasse vs geluid.")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=df_grootte, x="grootteklasse", y="SEL_dB", ax=ax)
        ax.set_title("Geluidsniveau per grootteklasse van vliegtuigen", fontsize=14)
        ax.set_xlabel("Grootteklasse", fontsize=12)
        ax.set_ylabel("Geluidsniveau (SEL_dB)", fontsize=12)
        st.pyplot(fig)

        st.markdown("""
        📊 Deze boxplot toont de verdeling van geluidsniveaus per grootteklasse.
        """)


# === 11. PAGINA: STIJGEND VS DALEND ===
elif keuze == "Stijgend vs dalend":
    st.header("🛫✈️ Stijgen vs. Landen: geluidsverdeling per richting")

    def bepaal_richting(altitude, distance):
        if pd.isna(altitude) or pd.isna(distance):
            return "Onbekend"
        if altitude > 1000 and distance < 20:
            return "Stijgend"
        elif altitude < 800 and distance > 20:
            return "Dalend"
        else:
            return "Onbekend"

    # Voeg toe aan df als nog niet bestaat
    if "vluchtrichting" not in df.columns:
        df["vluchtrichting"] = df.apply(
            lambda row: bepaal_richting(row["altitude"], row["distance"]), axis=1
        )

    df_richting = df[df["vluchtrichting"].isin(["Stijgend", "Dalend"])].dropna(subset=["SEL_dB"])
    df_stijgend = df_richting[df_richting["vluchtrichting"] == "Stijgend"]
    df_dalend = df_richting[df_richting["vluchtrichting"] == "Dalend"]

    aantal_stijgend = len(df_stijgend)
    aantal_dalend = len(df_dalend)

    st.markdown(f"""
    📊 **Aantal herkende vluchten**
    - 🛫 Stijgend: **{aantal_stijgend}**
    - 🛬 Dalend: **{aantal_dalend}**
    """)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Stijgend
    if aantal_stijgend > 0:
        sns.histplot(df_stijgend["SEL_dB"], bins=20, kde=True, ax=axes[0])
        axes[0].set_title("🛫 Stijgend", fontsize=13)
    else:
        axes[0].text(0.5, 0.5, "Geen data", ha="center", va="center", fontsize=12)
        axes[0].set_title("🛫 Stijgend (geen data)", fontsize=13)
    axes[0].set_xlabel("SEL_dB")
    axes[0].set_ylabel("Aantal metingen")

    # Dalend
    if aantal_dalend > 0:
        sns.histplot(df_dalend["SEL_dB"], bins=20, kde=True, ax=axes[1])
        axes[1].set_title("🛬 Dalend", fontsize=13)
    else:
        axes[1].text(0.5, 0.5, "Geen data", ha="center", va="center", fontsize=12)
        axes[1].set_title("🛬 Dalend (geen data)", fontsize=13)
    axes[1].set_xlabel("SEL_dB")

    fig.suptitle("🔊 Geluidsverdeling bij stijgende en dalende vluchten", fontsize=15)
    st.pyplot(fig)

    st.markdown("""
    - Links: **Stijgende vluchten**  
    - Rechts: **Dalende vluchten**  

    💡 Meestal geeft opstijgen (stijgend) meer geluidsoverlast.
    """)

