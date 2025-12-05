# UK Road Traffic AADF Prediction App

A machine learning model that predicts the Annual Average Daily Flow (AADF) of all motor vehicles on any counted point in the UK road network using official Department for Transport data.

Live app: https://mldacw116668-ga569qd2iyrarkepseappcj.streamlit.app/

What does it predict?

The estimated number of motor vehicles passing a road point every day on average over a full year (All Motor Vehicles AADF) — the same metric used by the UK government for road planning and funding allocation.

Features used
* Year (2000–2024)
* Region (e.g., London, Scotland, North West, etc.)
* Local Authority (e.g., Manchester City Council, Glasgow City, etc.)
* Road Category (Motorway, A-road, B-road, etc.)
* Road Type (Major / Minor)
* Estimation Method (Counted or Estimated)
* Direction of Travel (Northbound, Southbound, Eastbound, Westbound, Combined)

How to use the Streamlit app
* Open the app (run streamlit run app.py)
* Choose the year and road characteristics using the dropdowns and sliders
* Click "Predict Traffic"
* Get an instant prediction of the Annual Average Daily Flow (all motor vehicles)

Model details
* Trained on ~10,000 real UK road traffic count points (2000–2024) from the Department for Transport AADF dataset
* Best performing model: CatBoost – robust, fast, and handles one-hot encoded categories perfectly
* Preprocessing pipeline (ColumnTransformer + OneHotEncoder) saved together with the model using joblib
* All categorical features properly encoded, including mapping of local_authority_id → human-readable names

How to run locally
1. Clone or download this repository
2. Install the required packages:
    + pip install -r requirements.txt
3. Run the Streamlit app:
    + streamlit run app.py


Required libraries (requirements.txt)
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
streamlit


.
├── app.py                     
├── road_traffic.pkl            
├── requirements.txt          
├── ML_16668.ipynb             
├── data/road_traffic.csv                    
└── README.md                 


