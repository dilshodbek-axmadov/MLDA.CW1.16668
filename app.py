import streamlit as st
import pandas as pd
import joblib


# Load model and processor
saved_objects = joblib.load("road_traffic.pkl")
processor = saved_objects['preprocessor']
model = saved_objects['model']


# mapping local_authority_id with name
local_authority_map = {
    1: "Isles of Scilly",
    2: "Nottinghamshire",
    3: "Glasgow City",
    4: "North Lanarkshire",
    5: "Somerset",
    6: "Newport",
    7: "Bridgend",
    8: "Swansea",
    9: "Isle of Anglesey",
    10: "Gwynedd",
    11: "Conwy",
    12: "Denbighshire",
    13: "Monmouthshire",
    14: "Powys",
    15: "Carmarthenshire",
    16: "Pembrokeshire",
    17: "Neath Port Talbot",
    18: "The Vale of Glamorgan",
    19: "Cardiff",
    20: "Flintshire",
    21: "Merthyr Tydfil",
    22: "Rhondda, Cynon, Taff",
    23: "Ceredigion",
    24: "Blaenau Gwent",
    25: "Torfaen",
    26: "Wrexham",
    56: "Stockport",
    57: "Barnet",
    58: "Central Bedfordshire",
    59: "Northamptonshire",
    60: "Leicestershire",
    61: "Derbyshire",
    62: "Rotherham",
    64: "Medway",
    65: "Hampshire",
    66: "Hillingdon",
    67: "West Berkshire",
    68: "Wiltshire",
    69: "Worcestershire",
    70: "Gloucestershire",
    71: "Devon",
    72: "Warwickshire",
    73: "East Cheshire",
    76: "Lancashire",
    77: "Cumbria",
    78: "Hertfordshire",
    79: "Doncaster",
    80: "Kent",
    81: "West Sussex",
    83: "Buckinghamshire",
    84: "Wirral",
    89: "East Riding of Yorkshire",
    92: "Durham",
    93: "Tower Hamlets",
    94: "Sunderland",
    95: "Rochdale",
    97: "Cambridgeshire",
    99: "Lincolnshire",
    100: "North Yorkshire",
    101: "Gateshead",
    102: "Northumberland",
    105: "Greenwich",
    106: "Bexley",
    111: "Hounslow",
    115: "Bath and North East Somerset",
    116: "Shropshire",
    117: "Staffordshire",
    119: "Derby",
    121: "Enfield",
    122: "Hackney",
    123: "Essex",
    126: "Suffolk",
    127: "Southend-on-Sea",
    129: "Peterborough",
    130: "North Lincolnshire",
    133: "East Sussex",
    134: "Croydon",
    135: "Surrey",
    137: "Southampton",
    139: "Cornwall excluding Isles of Scilly",
    141: "Birmingham",
    142: "Oxfordshire",
    143: "South Gloucestershire",
    144: "Bristol, City of",
    148: "West Cheshire",
    150: "Walsall",
    154: "Norfolk",
    155: "Herefordshire, County of",
    159: "Sheffield",
    163: "Stockton-on-Tees",
    164: "Darlington",
    166: "South Tyneside",
    169: "Kingston upon Hull, City of",
    172: "Newcastle upon Tyne",
    175: "Richmond upon Thames",
    179: "Windsor and Maidenhead",
    181: "Swindon",
    183: "North Somerset",
    186: "Bedford",
    187: "Telford and Wrekin",
    189: "Dudley",
    192: "Sefton",
    194: "Barnsley",
    195: "Bradford",
    197: "Kirklees",
    199: "Isle of Wight",
    201: "Havering",
    202: "York",
    203: "Plymouth",
    209: "Bournemouth, Christchurch and Poole",
    210: "Dorset",
    211: "North Northamptonshire",
    212: "West Northamptonshire",
    213: "Cumberland",
    214: "Westmorland and Furness"
}

road_category_map = {
    "PA": "Class A Principal road",
    "TM": "M or Class A Trunk Motorway",
    "TA": "Class A Trunk road",
    "MB": "Class B road",
    "MCU": "Class C road or Unclassified road"
}

direction_map = {
    'W': "West",
    'E': "East",
    'N': "North",
    'S': "South",
    'C': "Combined"
}

region_names = ['South West', 'East Midlands', 'Scotland', 'Wales',
       'East of England', 'North West', 'North East',
       'Yorkshire and the Humber', 'South East', 'London',
       'West Midlands']

road_types = ['Major', 'Minor']
estimation_methods = ['Estimated', 'Counted']



st.title("Traffic Prediction App")

# Year
year = st.number_input("Year", min_value=2000, max_value=2024, value=2012, step=1)

# Region
region_name = st.selectbox("Region Name", region_names)

# Local Authority
local_authority_text = st.selectbox("Local Authority", list(local_authority_map.values()))
local_authority_id = [k for k, v in local_authority_map.items() if v == local_authority_text][0]

# Road Category (UI → key)
road_category_text = st.selectbox("Road Category", list(road_category_map.values()))
road_category = [k for k, v in road_category_map.items() if v == road_category_text][0]

# Road Type
road_type = st.selectbox("Road Type", road_types)

# Estimation Method
estimation_method = st.selectbox("Estimation Method", estimation_methods)

# Direction (UI → key)
direction_text = st.selectbox("Direction of Travel", list(direction_map.values()))
direction_of_travel = [k for k, v in direction_map.items() if v == direction_text][0]


# Predict
if st.button("Predict Traffic"):

    input_df = pd.DataFrame({
        "year": [year],
        "region_name": [region_name],
        "local_authority_id": [local_authority_id],
        "road_category": [road_category],          
        "road_type": [road_type],
        "estimation_method": [estimation_method],
        "direction_of_travel": [direction_of_travel]   
    })

    # Transform
    input_processed = processor.transform(input_df)

    # Predict
    prediction = model.predict(input_processed)

    st.success(f"Predicted All Motor Vehicles: {int(prediction[0])}")
