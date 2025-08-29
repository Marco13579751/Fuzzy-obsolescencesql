    import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import psycopg2
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import IPython.display as display
import datetime
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode


# --- PostgreSQL helpers ---
@st.cache_resource
def get_pg_conn():
    return psycopg2.connect(
        host=st.secrets["postgres"]["host"],
        port=st.secrets["postgres"].get("port", 5432),
        dbname=st.secrets["postgres"]["dbname"],
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"],
    )


def ensure_pg_tables():
    conn = get_pg_conn()
    with conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS devices (
                id SERIAL PRIMARY KEY,
                clinic TEXT NOT NULL,
                name TEXT NOT NULL,
                model TEXT,
                serial_number TEXT,
                purchase_date DATE,
                location TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS streamlit_valutazioni (
                id SERIAL PRIMARY KEY,
                user_email TEXT NOT NULL,
                device_id INTEGER REFERENCES devices(id) ON DELETE SET NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                normalized_age DOUBLE PRECISION,
                eq_function INTEGER,
                cost_levels DOUBLE PRECISION,
                failure_rate DOUBLE PRECISION,
                up_time DOUBLE PRECISION,
                reliability_score DOUBLE PRECISION,
                mission_score DOUBLE PRECISION,
                criticity_score DOUBLE PRECISION,
                obsolescenza DOUBLE PRECISION,
                parametri_json JSONB
            );
            CREATE INDEX IF NOT EXISTS idx_streamlit_val_user ON streamlit_valutazioni(user_email);
            CREATE INDEX IF NOT EXISTS idx_devices_clinic ON devices(clinic);
            """
        )
    return conn


def insert_device(clinic: str, name: str, model: str, serial_number: str, purchase_date: datetime.date | None, location: str | None):
    conn = ensure_pg_tables()
    with conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO devices (clinic, name, model, serial_number, purchase_date, location)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (clinic, name, model, serial_number, purchase_date, location),
        )
        row = cur.fetchone()
    return row[0] if row else None


def list_devices(clinic: str):
    conn = ensure_pg_tables()
    with conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, name, model, serial_number, purchase_date, location, created_at
            FROM devices
            WHERE clinic = %s
            ORDER BY created_at DESC
            """,
            (clinic,),
        )
        rows = cur.fetchall()
    devices = []
    for r in rows:
        devices.append({
            "id": r[0],
            "name": r[1],
            "model": r[2],
            "serial_number": r[3],
            "purchase_date": r[4],
            "location": r[5],
            "created_at": r[6],
        })
    return devices


def insert_valutazione(user_email: str, device_id: int | None, parametri: dict, scores: dict):
    conn = ensure_pg_tables()
    with conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO streamlit_valutazioni
            (user_email, device_id, normalized_age, eq_function, cost_levels, failure_rate, up_time,
             reliability_score, mission_score, criticity_score, obsolescenza, parametri_json)
            VALUES
            (%s, %s, %s, %s, %s, %s, %s,
             %s, %s, %s, %s, %s)
            """,
            (
                user_email,
                device_id,
                parametri.get("normalized_age"),
                parametri.get("eq_function"),
                parametri.get("cost_levels"),
                parametri.get("failure_rate"),
                parametri.get("up_time"),
                scores.get("reliability_score"),
                scores.get("mission_score"),
                scores.get("criticity_score"),
                scores.get("obsolescenza"),
                json.dumps(parametri),
            ),
        )


def load_valutazioni(user_email: str) -> list[dict]:
    conn = ensure_pg_tables()
    with conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT v.user_email, v.created_at,
                   v.normalized_age, v.eq_function, v.cost_levels, v.failure_rate, v.up_time,
                   v.reliability_score, v.mission_score, v.criticity_score, v.obsolescenza,
                   v.parametri_json, d.name as device_name, d.model as device_model
            FROM streamlit_valutazioni v
            LEFT JOIN devices d ON d.id = v.device_id
            WHERE v.user_email = %s
            ORDER BY v.created_at DESC
            """,
            (user_email,),
        )
        rows = cur.fetchall()

    results = []
    for r in rows:
        (
            _user,
            created_at,
            normalized_age,
            eq_function,
            cost_levels,
            failure_rate,
            up_time,
            reliability_score,
            mission_score,
            criticity_score,
            obsolescenza,
            parametri_json,
            device_name,
            device_model,
        ) = r

        base = {
            "normalized_age": normalized_age,
            "eq_function": eq_function,
            "cost_levels": cost_levels,
            "failure_rate": failure_rate,
            "up_time": up_time,
            "Reliability": reliability_score,
            "Mission": mission_score,
            "Criticity": criticity_score,
            "Obsolescence": obsolescenza,
            "Device": device_name,
            "Model": device_model,
            "created_at": created_at,
        }
        if parametri_json:
            try:
                base.update(json.loads(parametri_json))
            except Exception:
                pass
        results.append(base)
    return results


# --- Stato ---
if "user" not in st.session_state:
    st.session_state["user"] = None

# --- Identificazione clinica (niente Firebase) ---
st.title("Dashboard Obsolescence Medical Device")

with st.container():
    st.subheader("Clinic identification")
    clinic = st.text_input("Clinic/Hospital name", value=st.session_state.get("user") or "")
    if st.button("Set clinic"):
        st.session_state["user"] = clinic.strip()
        st.success("Clinic set")
        st.rerun()

if not st.session_state.get("user"):
    st.info("Set the clinic name to proceed.")
    st.stop()

user_email = st.session_state["user"]

# --- Gestione dispositivi ---
st.subheader("Devices registry")
with st.expander("Add new device"):
    colA, colB = st.columns(2)
    with colA:
        dev_name = st.text_input("Device name")
        dev_model = st.text_input("Model")
        dev_serial = st.text_input("Serial number")
    with colB:
        dev_purchase = st.date_input("Purchase date", value=None)
        dev_location = st.text_input("Location/Department")
    if st.button("Save device"):
        if not dev_name:
            st.error("Device name is required")
        else:
            try:
                insert_device(user_email, dev_name, dev_model, dev_serial, dev_purchase, dev_location)
                st.success("Device saved")
                st.rerun()
            except Exception as e:
                st.error(f"Errore salvataggio device: {e}")

devices = list_devices(user_email)
if devices:
    df_devices = pd.DataFrame(devices)
    st.dataframe(df_devices)
else:
    st.write("No devices yet.")

# --- Selezione device per valutazione ---
selected_device_id = None
if devices:
    options = {f"{d['name']} ({d.get('model') or ''})": d["id"] for d in devices}
    sel = st.selectbox("Select device for evaluation", options=list(options.keys()))
    selected_device_id = options[sel]

# --- Input utente con 5 parametri ---
st.subheader("📥 Add Device's informations (parameters)")

parametri_nome = [
    'normalizedAge', 'normalizedRiskLevels', 'normalizedfunctionLevels',
    'normalizedStateLevels', 'normalizedLifeResLevels', 'normalizedObsLevels',
    'normalizedUtilizationLevels', 'normalizedUptime',
    'normalizedfaultRateLevels', 'normalizedEoLS'
]
parametri_nome_prova_con_2_parametri=['normalized_age','eq_function','cost_levels','failure_rate','up_time']

inputs = []

if st.button("🔄 Clear cache & refresh"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()
    
# Mostra i box in righe da 3 colonne
colonne = st.columns(3)

for i, nome in enumerate(parametri_nome_prova_con_2_parametri):
    col = colonne[i % 3]
    with col:
        if nome == "normalized_age":
            data_acquisto = st.date_input("Date of purchase")
            oggi = datetime.date.today()
            eta_giorni = (oggi - data_acquisto).days if data_acquisto else 0
            eta = eta_giorni / 365 if data_acquisto else 0.0
            val = eta
            st.write(f"Age: {eta:.2f}")

        elif nome == "eq_function":
            val = st.selectbox(
                "Equipment function",
                options=[1, 2, 3, 4],
                key=f"failure_rate_{i}"
            )
        elif nome=="cost_levels" :
            val = st.number_input(
                "Cost",
                min_value=0.0,
                step=1.0,
                format="%.2f",
                key=f"cost_{i}"
            )
        elif nome=="up_time" :
            val = st.number_input(
                "Uptime",
                min_value=0.0,
                step=1.0,
                format="%.2f",
                key=f"up_time_{i}"
            )
        elif nome=="failure_rate" :
            val = st.number_input(
                "Failure rate",
                min_value=0.0,
                step=1.0,
                format="%.2f",
                key=f"failure_{i}"
            )
        inputs.append(val if val != 0.0 else None)


# --- Fuzzy logic ---
normalized_age = ctrl.Antecedent(np.arange(0, 11, 0.1), 'normalized_age')
failure_rate=ctrl.Antecedent(np.arange(0, 1, 0.1), 'failure_rate')

eq_function = ctrl.Antecedent(np.arange(0, 4, 0.01), 'eq_function')
up_time=ctrl.Antecedent(np.arange(0,36,0.01),'up_time')

cost_levels=ctrl.Antecedent(np.arange(0,1001,1),'cost_levels')
    
#Categorie madre
reliability=ctrl.Consequent(np.arange(0,10.1, 0.01), 'reliability')
mission=ctrl.Consequent(np.arange(0,10.1, 0.01), 'mission')
reliability_result = ctrl.Antecedent(np.arange(0, 10.1, 0.01), 'reliability_result')
mission_result = ctrl.Antecedent(np.arange(0, 10.1, 0.01), 'mission_result')

# Add output variable (Consequent)
criticity = ctrl.Consequent(np.arange(0, 10.1, 0.01), 'criticity')


# Define membership functions for normalizedAge

normalized_age['New'] = fuzz.trapmf(normalized_age.universe, [0, 0, 2, 5])
normalized_age['Middle'] = fuzz.trimf(normalized_age.universe, [3, 5, 7])
normalized_age['Old'] = fuzz.trapmf(normalized_age.universe, [5, 8, 10, 10])

failure_rate['Low'] = fuzz.trimf(failure_rate.universe, [0, 0.20,0.40])
failure_rate['Medium'] = fuzz.trimf(failure_rate.universe, [0.20,0.50,0.80])
failure_rate['High'] = fuzz.trimf(failure_rate.universe, [0.60, 0.80, 1])

# Define membership functions for normalizedfaultRateLevels

eq_function['Under trh'] = fuzz.gaussmf(eq_function.universe, 1, 0.1)
eq_function['Around trh'] = fuzz.gaussmf(eq_function.universe, 2, 0.1)
eq_function['Above trh'] = fuzz.gaussmf(eq_function.universe, 3, 0.1)

up_time['Low'] = fuzz.trapmf(up_time.universe, [0,0,8,16])
up_time['Middle'] = fuzz.trimf(up_time.universe, [8,18,28])
up_time['High'] = fuzz.trapmf(up_time.universe, [20,28,36,36])

# Define membership functions for Costlevels
cost_levels['low']=fuzz.trapmf(cost_levels.universe, [0,0,200,500])
cost_levels['medium']=fuzz.trimf(cost_levels.universe, [300,500,700])
cost_levels['high']=fuzz.trapmf(cost_levels.universe, [500,800,1000,1000])

#Membership madre consequent
reliability['Low']=fuzz.trapmf(reliability.universe, [0,0,2,5])
reliability['Medium']=fuzz.trimf(reliability.universe, [3,5,7])
reliability['High']=fuzz.trapmf(reliability.universe, [5,8,10,10])

mission['Low']=fuzz.trapmf(mission.universe, [0,0,2,5])
mission['Medium']=fuzz.trimf(mission.universe, [3,5,7])
mission['High']=fuzz.trapmf(mission.universe, [5,8,10,10])

# Membershipmadre antecedente 
reliability_result['Low'] = fuzz.trapmf(reliability_result.universe, [0, 0, 2, 5])
reliability_result['Medium'] = fuzz.trimf(reliability_result.universe, [3, 5, 7])
reliability_result['High'] = fuzz.trapmf(reliability_result.universe, [5, 8, 10, 10])

mission_result['Low'] = fuzz.trapmf(mission_result.universe, [0, 0, 2, 5])
mission_result['Medium'] = fuzz.trimf(mission_result.universe, [3, 5, 7])
mission_result['High'] = fuzz.trapmf(mission_result.universe, [5, 8, 10, 10])


# Define membership functions for Criticity
criticity['VeryLow'] = fuzz.gaussmf(criticity.universe, 1, 0.7)
criticity['Low'] = fuzz.gaussmf(criticity.universe, 3, 0.7)
criticity['Medium'] = fuzz.gaussmf(criticity.universe, 5, 0.7)
criticity['High'] = fuzz.gaussmf(criticity.universe, 7, 0.7)
criticity['VeryHigh'] = fuzz.gaussmf(criticity.universe, 9, 0.7)


# Define fuzzy rules

rule_r=[
    ctrl.Rule(failure_rate['High'] & normalized_age['New'], reliability['Medium']),
    ctrl.Rule(failure_rate['High'] & normalized_age['Middle'], reliability['Medium']),
    ctrl.Rule(failure_rate['High'] & normalized_age['Old'], reliability['High']),

    ctrl.Rule(failure_rate['Medium'] & normalized_age['New'], reliability['Low']),
    ctrl.Rule(failure_rate['Medium'] & normalized_age['Middle'], reliability['Medium']),
    ctrl.Rule(failure_rate['Medium'] & normalized_age['Old'], reliability['High']),

    ctrl.Rule(failure_rate['Low'] & normalized_age['New'], reliability['Low']),
    ctrl.Rule(failure_rate['Low'] & normalized_age['Middle'], reliability['Low']),
    ctrl.Rule(failure_rate['Low'] & normalized_age['Old'], reliability['Medium']),
]

rule_m=[
    ctrl.Rule(eq_function['Above trh'] & up_time['Low'], mission['Medium']),
    ctrl.Rule(eq_function['Above trh'] & up_time['Middle'], mission['High']),
    ctrl.Rule(eq_function['Above trh'] & up_time['High'], mission['High']),

    ctrl.Rule(eq_function['Around trh'] & up_time['Low'], mission['Low']),
    ctrl.Rule(eq_function['Around trh'] & up_time['Middle'], mission['Medium']),
    ctrl.Rule(eq_function['Around trh'] & up_time['High'], mission['High']),

    ctrl.Rule(eq_function['Under trh'] & up_time['Low'], mission['Low']),
    ctrl.Rule(eq_function['Under trh'] & up_time['Middle'], mission['Medium']),
    ctrl.Rule(eq_function['Under trh'] & up_time['High'], mission['High']),
]


rules = [
    ctrl.Rule(normalized_age['New'] & eq_function['Under trh'], criticity['VeryLow']),
    ctrl.Rule(normalized_age['New'] & eq_function['Around trh'], criticity['Low']),
    ctrl.Rule(normalized_age['New'] & eq_function['Above trh'], criticity['Medium']),

    ctrl.Rule(normalized_age['Middle'] & eq_function['Under trh'], criticity['Low']),
    ctrl.Rule(normalized_age['Middle'] & eq_function['Around trh'], criticity['Medium']),
    ctrl.Rule(normalized_age['Middle'] & eq_function['Above trh'], criticity['High']),

    ctrl.Rule(normalized_age['Old'] & eq_function['Under trh'], criticity['Low']),
    ctrl.Rule(normalized_age['Old'] & eq_function['Around trh'], criticity['High']),
    ctrl.Rule(normalized_age['Old'] & eq_function['Above trh'], criticity['VeryHigh']),

    ctrl.Rule(cost_levels['high'] & normalized_age['New'], criticity['Low']),
    ctrl.Rule(cost_levels['high'] & normalized_age['Middle'], criticity['Medium']),
    ctrl.Rule(cost_levels['high'] & normalized_age['Old'], criticity['VeryHigh']),

    ctrl.Rule(cost_levels['medium'] & normalized_age['New'], criticity['VeryLow']),
    ctrl.Rule(cost_levels['medium'] & normalized_age['Middle'], criticity['Medium']),
    ctrl.Rule(cost_levels['medium'] & normalized_age['Old'], criticity['High']),

    ctrl.Rule(cost_levels['low'] & normalized_age['New'], criticity['VeryLow']),
    ctrl.Rule(cost_levels['low'] & normalized_age['Middle'], criticity['Low']),
    ctrl.Rule(cost_levels['low'] & normalized_age['Old'], criticity['Medium']),

    ctrl.Rule(cost_levels['high'] & eq_function['Under trh'], criticity['Low']),
    ctrl.Rule(cost_levels['high'] & eq_function['Around trh'], criticity['High']),
    ctrl.Rule(cost_levels['high'] & eq_function['Above trh'], criticity['VeryHigh']),

    ctrl.Rule(cost_levels['medium'] & eq_function['Under trh'], criticity['VeryLow']),
    ctrl.Rule(cost_levels['medium'] & eq_function['Around trh'], criticity['Medium']),
    ctrl.Rule(cost_levels['medium'] & eq_function['Above trh'], criticity['High']),

    ctrl.Rule(cost_levels['low'] & eq_function['Under trh'], criticity['VeryLow']),
    ctrl.Rule(cost_levels['low'] & eq_function['Around trh'], criticity['Low']),
    ctrl.Rule(cost_levels['low'] & eq_function['Above trh'], criticity['Medium']),
]

# Create the control system
mission_ctrl=ctrl.ControlSystem(rule_m)
reliability_ctrl=ctrl.ControlSystem(rule_r)

plt.style.use("seaborn-v0_8-muted")

def plot_membership_functions(antecedent, title):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(antecedent.terms)))

    for idx, term in enumerate(antecedent.terms):
        ax.plot(
            antecedent.universe,
            antecedent[term].mf,
            label=term.capitalize(),
            linewidth=1,
            color=colors[idx]
        )

    ax.set_title(title, fontsize=9, weight='bold', pad=10)
    ax.set_xlabel("Valore", fontsize=6)
    ax.set_ylabel("Appartenenza", fontsize=6)
    ax.tick_params(labelsize=6)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25),
              ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

# Simulators
mission_simulation=ctrl.ControlSystemSimulation(mission_ctrl)
reliability_simulation=ctrl.ControlSystemSimulation(reliability_ctrl)

rule_f = [
    ctrl.Rule(mission_result['High'] & reliability_result['High'], criticity['VeryHigh']),
    ctrl.Rule(mission_result['High'] & reliability_result['Medium'], criticity['High']),
    ctrl.Rule(mission_result['High'] & reliability_result['Low'], criticity['High']),

    ctrl.Rule(mission_result['Medium'] & reliability_result['High'], criticity['VeryHigh']),
    ctrl.Rule(mission_result['Medium'] & reliability_result['Medium'], criticity['Medium']),
    ctrl.Rule(mission_result['Medium'] & reliability_result['Low'], criticity['Low']),

    ctrl.Rule(mission_result['Low'] & reliability_result['High'], criticity['High']),
    ctrl.Rule(mission_result['Low'] & reliability_result['Medium'], criticity['Low']),
    ctrl.Rule(mission_result['Low'] & reliability_result['Low'], criticity['VeryLow']),
]

# Compute inputs
for nome, val in zip(parametri_nome_prova_con_2_parametri, inputs):
    valore = val if val is not None else 0.0
    if nome in ["up_time", "eq_function"]:
        mission_simulation.input[nome] = valore
    elif nome in ["normalized_age", "failure_rate"]:
        reliability_simulation.input[nome] = valore


def show_fuzzy_output(fuzzy_var, sim):
    sim.compute()
    var_name = fuzzy_var.label
    output_keys = list(sim.output.keys())
    matching_key = None
    for k in output_keys:
        if k.lower() == var_name.lower():
            matching_key = k
            break
    if matching_key is None:
        raise KeyError(f"Variabile '{var_name}' non trovata tra le uscite disponibili: {output_keys}")
    output_value = sim.output[matching_key]

    fig, ax = plt.subplots(figsize=(5, 2.5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(fuzzy_var.terms)))
    x = fuzzy_var.universe

    for idx, term in enumerate(fuzzy_var.terms):
        mf = fuzzy_var[term].mf
        y = mf
        ax.plot(x, y, label=term.capitalize(), linewidth=1, color=colors[idx])
        activation = fuzz.interp_membership(x, y, output_value)
        ax.fill_between(x, 0, np.fmin(activation, y), alpha=0.4, color=colors[idx])

    ax.axvline(x=output_value, color='red', linestyle='--', linewidth=1,
               label=f'Uscita = {output_value:.2f}')

    ax.set_title(f"Output fuzzy: {matching_key}", fontsize=9, weight='bold', pad=10)
    ax.set_xlabel("Valore", fontsize=6)
    ax.set_ylabel("Appartenenza", fontsize=6)
    ax.tick_params(labelsize=6)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25),
              ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    return output_value

criticity_ctrl = ctrl.ControlSystem(rule_f)
criticity_simulation = ctrl.ControlSystemSimulation(criticity_ctrl)

reliability_score=show_fuzzy_output(reliability, reliability_simulation)
mission_score=show_fuzzy_output(mission, mission_simulation)

criticity_simulation.input['mission_result'] = mission_score
criticity_simulation.input['reliability_result'] = reliability_score

criticity_score=show_fuzzy_output(criticity, criticity_simulation)

# Store the result (scaled by 10 as in your Matlab code)
obsolescenza = criticity_simulation.output['criticity'] * 10

if obsolescenza is not None:
    st.write("**Obsolescence score:**", f"{obsolescenza:.2f}")
    if obsolescenza > 60:
        st.error("⚠️ Device partially obsolet")
    else:
        st.success("✅ Device in good condition")
else:
    st.info("🟡 Inserisci almeno un parametro per calcolare lo score")


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

plot_membership_functions(normalized_age, 'Age')
plot_membership_functions(eq_function, 'Failure rate')
plot_membership_functions(cost_levels, 'Cost')
plot_membership_functions(failure_rate, 'Failure rate')
plot_membership_functions(up_time, 'Uptime')

# --- Salvataggio su PostgreSQL ---
if st.button("Save valuation"):
    parametri_dict = {
        nome: val if val is not None else None
        for nome, val in zip(parametri_nome_prova_con_2_parametri, inputs)
    }
    scores = {
        "reliability_score": float(reliability_score) if reliability_score is not None else None,
        "mission_score": float(mission_score) if mission_score is not None else None,
        "criticity_score": float(criticity_score) if criticity_score is not None else None,
        "obsolescenza": float(f"{obsolescenza:.2f}") if obsolescenza is not None else None,
    }
    try:
        insert_valutazione(user_email, selected_device_id, parametri_dict, scores)
        st.success("✅ Valuation saved to PostgreSQL!")
    except Exception as e:
        st.error(f"Errore salvataggio PostgreSQL: {e}")

st.subheader("📋 Valutations saved")
try:
    valutazioni_list = load_valutazioni(user_email)
except Exception as e:
    st.error(f"Errore lettura PostgreSQL: {e}")
    valutazioni_list = []

rows = []
for d in valutazioni_list:
    row = {k: safe_float(v) for k, v in d.items() if k not in ("created_at", "Device", "Model")}
    row["Device"] = d.get("Device")
    row["Model"] = d.get("Model")
    rows.append(row)

df = pd.DataFrame(rows)

gb = GridOptionsBuilder.from_dataframe(df)

gb.configure_default_column(
    editable=True, 
    resizable=True, 
    sortable=True,
    filter=True,
    floatingFilter=False
)

for col in df.columns:
    if col != "Obsolescence":
        gb.configure_column(
            col, 
            editable=True,
            type=["numericColumn", "numberColumnFilter", "customNumericFormat"],
            precision=2
        )

gb.configure_column(
    "Obsolescence", 
    editable=False,
    cellStyle={'backgroundColor': '#f0f0f0'}
)

gb.configure_grid_options(
    enableRangeSelection=True,
    enableClipboard=True,
    suppressMovableColumns=False
)

grid_options = gb.build()

st.write("### Valutazioni (Clicca doppio click sulle celle per modificare)")

grid_response = AgGrid(
    df,
    gridOptions=grid_options,
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    update_mode=GridUpdateMode.MODEL_CHANGED,
    fit_columns_on_grid_load=True,
    enable_enterprise_modules=False,
    allow_unsafe_jscode=True,
    height=400,
    width='100%',
    reload_data=True,
)

df_edited = grid_response['data']

if not df.equals(df_edited):
    st.write("### Dati modificati")
    st.dataframe(df_edited)
    if st.button("Salva modifiche"):
        st.success("Modifiche salvate!")
else:
    st.write("### Nessuna modifica effettuata")
