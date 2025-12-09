import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
import os
from streamlit_gsheets import GSheetsConnection
# --- 設定頁面配置 ---
st.set_page_config(
    page_title="溫室環境決策系統 V6.0 (Python版)",
    page_icon="🌿",
    layout="wide"
)

# ==========================================
# 1. 核心工具函式庫 (讀取 CSV 與 自動掃描)
# ==========================================

def scan_and_load_weather_data(base_folder='weather_data'):
    """
    掃描 weather_data 資料夾
    - 支援 CWA 月報表 (12列): 直接讀取
    - 支援 CWA 時報表 (8760列): 自動統計 (日射量累加, 溫度取極值)
    """
    loaded_locations = {}
    
    if not os.path.exists(base_folder):
        st.sidebar.warning(f"⚠️ 找不到 '{base_folder}' 資料夾")
        return {}

    files = [f for f in os.listdir(base_folder) if f.endswith('.csv')]
    
    if not files:
        st.sidebar.info(f"📂 '{base_folder}' 是空的")
        return {}

    for f in files:
        path = os.path.join(base_folder, f)
        try:
            # 1. 嘗試讀取測站名稱
            station_name = f.split('.')[0]
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                    first_line = file.readline()
                    if '測站' in first_line:
                        parts = first_line.split(',')
                        if len(parts) > 1: station_name = parts[1].strip()
            except: pass

            # 2. [強化讀取] 加入 on_bad_lines='skip' 防止格式錯誤
            try: df = pd.read_csv(path, header=1, encoding='utf-8', on_bad_lines='skip')
            except: 
                try: df = pd.read_csv(path, header=1, encoding='big5', on_bad_lines='skip')
                except: df = pd.read_csv(path, header=0, encoding='utf-8', on_bad_lines='skip')

            df.columns = [c.strip() for c in df.columns]
            
            # 3. 智慧欄位對照 (支援各種寫法)
            col_map = {}
            for c in df.columns:
                if '時間' in c or 'Time' in c: col_map['time'] = c
                elif '氣溫' in c or 'Temp' in c: col_map['temp'] = c
                elif '濕度' in c or 'RH' in c: col_map['rh'] = c
                elif '風速' in c or 'Wind' in c: col_map['wind'] = c
                elif '日射' in c or 'Solar' in c: col_map['solar'] = c
    

            if 'time' not in col_map: continue 

            # 4. 處理時間與數值
            df['Date'] = pd.to_datetime(df[col_map['time']], errors='coerce')
            df = df.dropna(subset=['Date'])
            df['Month'] = df['Date'].dt.month
            
            # ... (前面是日期處理) ...
            
            # [修正] 處理數值 (轉 float，但保留 NaN 以免影響平均值)
            for k, col in col_map.items():
                if k != 'time':
                    # coerce 會把無法轉數字的變成 NaN，我們保留 NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce') 

            # ==========================================
            # [核心] 判斷是「月資料」還是「時資料」
            # ==========================================
            data_dict = {
                'months': list(range(1, 13)),
                'temps': [], 'maxTemps': [], 'minTemps': [],
                'humidities': [], 'solar': [], 'wind': [],
                'marketPrice': [30]*12
            }

            # --- 情況 A: 資料量少 (月報表) ---
            if len(df) <= 24:
                monthly_grp = df.groupby('Month')
                for m in range(1, 13):
                    if m in monthly_grp.groups:
                        g = monthly_grp.get_group(m)
                        # [修正] 使用 mean() 會自動忽略 NaN，不會被 0 拉低
                        data_dict['temps'].append(float(g[col_map['temp']].mean()))
                        
                        max_col = next((c for c in df.columns if '最高' in c and '溫' in c), col_map['temp'])
                        min_col = next((c for c in df.columns if '最低' in c and '溫' in c), col_map['temp'])
                        data_dict['maxTemps'].append(float(g[max_col].max()))
                        data_dict['minTemps'].append(float(g[min_col].min()))
                        
                        # 濕度與風速
                        rh_val = g[col_map.get('rh', col_map['temp'])].mean()
                        data_dict['humidities'].append(float(rh_val) if not pd.isna(rh_val) else 75.0)
                        
                        wind_val = g[col_map.get('wind', col_map['temp'])].mean()
                        data_dict['wind'].append(float(wind_val) if not pd.isna(wind_val) else 1.0)
                        
                        # 日射量
                        if 'solar' in col_map:
                            val = g[col_map['solar']].mean()
                            if val > 50: val /= 30 
                            data_dict['solar'].append(float(val) if not pd.isna(val) else 12.0)
                        else:
                            data_dict['solar'].append(12.0)
                    else:
                        # 該月完全無資料才補預設值
                        data_dict['temps'].append(25.0); data_dict['maxTemps'].append(30.0); data_dict['minTemps'].append(20.0)
                        data_dict['humidities'].append(75.0); data_dict['solar'].append(12.0); data_dict['wind'].append(1.0)

            # --- 情況 B: 資料量大 (時報表) ---
            else:
                for m in range(1, 13):
                    g = df[df['Month'] == m]
                    if not g.empty:
                        # [修正] 這裡也一樣，直接 mean() 忽略 NaN
                        data_dict['temps'].append(float(g[col_map['temp']].mean()))
                        data_dict['maxTemps'].append(float(g[col_map['temp']].max()))
                        data_dict['minTemps'].append(float(g[col_map['temp']].min()))
                        
                        if 'rh' in col_map: 
                            val = g[col_map['rh']].mean()
                            data_dict['humidities'].append(float(val) if not pd.isna(val) else 75.0)
                        else: data_dict['humidities'].append(75.0)
                        
                        if 'wind' in col_map: 
                            val = g[col_map['wind']].mean()
                            data_dict['wind'].append(float(val) if not pd.isna(val) else 1.0)
                        else: data_dict['wind'].append(1.0)
                        
                        if 'solar' in col_map:
                            # 日射量：NaN 視為 0 (晚上或儀器故障算沒光) 比較合理
                            g_solar = g[col_map['solar']].fillna(0)
                            daily_sums = g.groupby(g['Date'].dt.date)[col_map['solar']].sum()
                            avg_daily_solar = daily_sums.mean()
                            data_dict['solar'].append(float(avg_daily_solar))
                        else:
                            data_dict['solar'].append(12.0)
                    else:
                        data_dict['temps'].append(25.0); data_dict['maxTemps'].append(30.0); data_dict['minTemps'].append(20.0)
                        data_dict['humidities'].append(75.0); data_dict['solar'].append(12.0); data_dict['wind'].append(1.0)

            

            # 5. 存入 Locations
            station_id = f.split('.')[0]
            desc = '時報表統計數據' if len(df) > 24 else '月報表數據'
            
            loaded_locations[station_id] = {
                'id': station_id,
                'name': f"{station_name}",
                'description': f'{desc} (來自 {f})',
                'data': data_dict
            }
            
            # 顯示於側邊欄，不彈出
            st.sidebar.success(f"✅ {station_name} 載入成功")
            
        except Exception as e:
            st.sidebar.error(f"❌ {f} 讀取失敗: {e}")
            continue

    return loaded_locations

def scan_and_load_market_prices(base_folder='market_data'):
    """掃描 market_data 資料夾內的 CSV"""
    price_db = {}
    if not os.path.exists(base_folder): return {}
    files = [f for f in os.listdir(base_folder) if f.endswith('.csv')]
    for f in files:
        try:
            path = os.path.join(base_folder, f)
            # 嘗試讀取，自動偵測 header
            try: df = pd.read_csv(path, header=2)
            except: df = pd.read_csv(path, header=0) # 備案

            if '交易日期' in df.columns and '平均價' in df.columns:
                df['M'] = df['交易日期'].astype(str).apply(lambda x: int(x.split('年')[1].replace('月','')) if '年' in x else None)
                monthly_avg = df.groupby('M')['平均價'].mean()
                price_list = [round(monthly_avg.get(m, 30.0), 1) for m in range(1, 13)]
                name = os.path.splitext(f)[0]
                price_db[name] = price_list
        except: continue
    return price_db

def load_fan_database(folder='equipment_data', filename='greenhouse_fans.csv', category_filter=None):
    paths = [os.path.join(folder, filename), filename]
    path = None
    for p in paths:
        if os.path.exists(p):
            path = p
            break
    if path:
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip() for c in df.columns]
            if category_filter:
                df = df[df['Category'].str.contains(category_filter, case=False, na=False)]
            
            def parse_dist(val):
                if pd.isna(val): return 0.0
                s = str(val).strip()
                try:
                    if '-' in s: parts = s.split('-'); return (float(parts[0]) + float(parts[1])) / 2
                    elif '~' in s: parts = s.split('~'); return (float(parts[0]) + float(parts[1])) / 2
                    else: return float(s)
                except: return 0.0

            if 'Throw_Distance_m' in df.columns:
                df['Throw_Distance_m'] = df['Throw_Distance_m'].apply(parse_dist)
            else: df['Throw_Distance_m'] = 0.0

            df['Label'] = ("[" + df['Model'].astype(str) + "] " + df['Description'].astype(str) + " (" + df['Diameter_Inch'].astype(str) + "吋 | " + pd.to_numeric(df['Airflow_CMH'], errors='coerce').fillna(0).apply(lambda x: f"{x:,.0f}") + " CMH)")
            for col in ['Airflow_CMH', 'Power_W', 'Price_NTD']: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            return df
        except: return pd.DataFrame()
    return pd.DataFrame()

def load_net_database(folder='equipment_data', filename='insect_nets.csv'):
    paths = [os.path.join(folder, filename), filename]
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                df['Label'] = df['Mesh'].astype(str) + "目 - " + df['Description']
                return df
            except: return pd.DataFrame()
    return pd.DataFrame()

def load_mat_database(folder='equipment_data', filename='greenhouse_materials.csv'):
    paths = [os.path.join(folder, filename), filename]
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                df['Label'] = df['Material_Code'] + " - " + df['Material_Type'] + " (" + df['Light_Property'] + ")"
                return df
            except: return pd.DataFrame()
    return pd.DataFrame()

def load_fog_database(folder='equipment_data', filename='foggingsystem.csv'):
    paths = [os.path.join(folder, filename), filename]
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                df['Label'] = df['Spray_Capacity_g_m2_hr'].astype(str) + " g/m²/hr (降溫後>30°C剩 " + df['Hours_Air_Temp_gt_30C'].astype(str) + "hr)"
                return df
            except: return pd.DataFrame()
    return pd.DataFrame()

# ==========================================
# 2. 資料庫初始化
# ==========================================

BUILTIN_LOCATIONS = {
    'pingtung': {
        'id': 'pingtung', 'name': '氣候範例',
        'description': '熱帶季風氣候，日射量高。',
        'data': {
            'months': list(range(1, 13)),
            'temps': [19.8, 20.6, 23.2, 26.0, 27.5, 29.0, 29.2, 28.3, 28.1, 26.6, 24.8, 20.8],
            'maxTemps': [25.5, 26.8, 29.5, 31.8, 33.2, 34.1, 34.5, 34.0, 33.6, 31.5, 29.2, 26.5],
            'minTemps': [15.2, 16.0, 18.2, 21.8, 24.2, 25.8, 25.9, 25.6, 24.8, 23.2, 20.8, 16.8],
            'humidities': [72, 73, 72, 74, 76, 78, 80, 82, 78, 74, 72, 71],
            'solar': [9.5, 10.5, 12.8, 14.5, 15.8, 16.2, 16.5, 15.0, 14.2, 12.5, 10.5, 9.0],
            'wind': [1.0, 1.2, 1.1, 1.1, 1.0, 1.2, 1.3, 1.2, 1.0, 0.9, 0.8, 0.9],
            'marketPrice': [35, 28, 25, 22, 20, 35, 45, 48, 42, 38, 30, 32]
        }
    }
}

CROP_DATABASE = {
    'lettuce': {'id': 'lettuce', 'name': '萵苣', 'idealTemp': 20, 'tempTolerance': 6, 'baseWeight': 0.35, 'cycleDays': 45, 'lightSaturation': 11, 'lightSlope': 1.2, 'price': 45},
    'cabbage': {'id': 'cabbage', 'name': '小白菜', 'idealTemp': 25, 'tempTolerance': 8, 'baseWeight': 0.15, 'cycleDays': 28, 'lightSaturation': 10, 'lightSlope': 0.8, 'price': 35},
    'spinach': {'id': 'spinach', 'name': '蕹菜', 'idealTemp': 28, 'tempTolerance': 10, 'baseWeight': 0.25, 'cycleDays': 25, 'lightSaturation': 12, 'lightSlope': 1.0, 'price': 30},
    'tomato': {'id': 'tomato', 'name': '小番茄', 'idealTemp': 23, 'tempTolerance': 8, 'baseWeight': 0.6, 'cycleDays': 60, 'lightSaturation': 14, 'lightSlope': 1.5, 'price': 120}
}

MATERIAL_OPTIONS = {
    'glass': {'label': '散射玻璃 (Glass)', 'uValue': 5.8, 'trans': 0.9},
    'poly': {'label': '塑膠薄膜 (Poly)', 'uValue': 6.0, 'trans': 0.85},
}

ROOF_OPTIONS = {'venlo': 'Venlo式', 'tunnel': '圓頂隧道式', 'single slope': '單斜屋頂式'}

# ==========================================
# 3. 核心運算邏輯
# ==========================================

def run_simulation(target_gh_specs, target_fan_specs, target_climate, monthly_crops, planting_density, annual_cycles, market_prices):
    floor_area = target_gh_specs['width'] * target_gh_specs['length']
    vol_coef = target_gh_specs.get('_vol_coef', 1.2)
    surf_coef = target_gh_specs.get('_surf_coef', 1.15)
    vent_eff = target_gh_specs.get('_vent_eff', 1.0)
    
    volume = floor_area * target_gh_specs['gutterHeight'] * vol_coef
    wall_area = 2 * (target_gh_specs['width'] + target_gh_specs['length']) * target_gh_specs['gutterHeight']
    surface_area = (floor_area * surf_coef) + wall_area
    cross_section_area = target_gh_specs['width'] * target_gh_specs['gutterHeight']
    planting_area = floor_area * 0.6 
    
    selected_mat = MATERIAL_OPTIONS.get(target_gh_specs['material'], MATERIAL_OPTIONS['glass'])
    u_value = selected_mat['uValue']
    transmissivity = selected_mat['trans']

    data = []
    total_revenue = 0
    total_yield = 0
    max_summer_temp = 0

    for i in range(12):
        crop_id = monthly_crops[i]
        crop = CROP_DATABASE.get(crop_id, CROP_DATABASE['lettuce'])

        t_out = target_climate['temps'][i]
        rh_out = target_climate['humidities'][i]
        solar_out = target_climate['solar'][i]
        wind_speed = target_climate['wind'][i]

        shading_factor = target_gh_specs['shadingScreen'] / 100
        t_trans = transmissivity * (1 - shading_factor)
        
        q_solar = (solar_out * 1000000 / 43200) * floor_area * t_trans
        vent_area = target_gh_specs['roofVentArea'] + target_gh_specs['sideVentArea']
        net_permeability = target_gh_specs['insectNet'] / 100
        natural_vent_rate = wind_speed * vent_area * 0.4 * net_permeability * vent_eff
        forced_vent_rate = (target_fan_specs['exhaustCount'] * target_fan_specs['exhaustFlow']) / 3600
        total_vent_rate = natural_vent_rate + forced_vent_rate
        
        if volume == 0: ach = 0
        else: ach = (total_vent_rate * 3600) / volume
        
        q_vent = total_vent_rate * 1200
        q_cond = u_value * surface_area
        denom = q_vent + q_cond
        delta_t = q_solar / denom if denom > 0 else 0
        t_in = t_out + delta_t
        
        if i == 6: max_summer_temp = t_in 

        # 基準情境計算
        delta_t_base = delta_t * 1.5 
        t_in_base = t_out + delta_t_base

        transpiration_factor = 20
        moisture_accumulation = transpiration_factor / (ach * 0.5 + 1)
        rh_in = min(98, max(40, rh_out + moisture_accumulation))

        v_thermal = 0.03 * math.sqrt(max(0, delta_t) * target_gh_specs['gutterHeight'])
        v_exhaust = forced_vent_rate / cross_section_area if cross_section_area > 0 else 0
        v_circ = (target_fan_specs['circCount'] * 0.05) * (1500 / floor_area) if floor_area > 0 else 0
        v_in = v_thermal + v_exhaust + v_circ

        # 高溫時數積分
        heat_hours30_base = 0
        heat_hours35_base = 0
        heat_hours30_in = 0
        heat_hours35_in = 0
        
        for h in range(24):
            hour_angle = (h - 9) * (math.pi / 12)
            temp_var = 5 
            temp_now_base = t_in_base + temp_var * math.sin(hour_angle)
            if temp_now_base >= 30: heat_hours30_base += 1
            if temp_now_base >= 35: heat_hours35_base += 1

            temp_now_in = t_in + temp_var * math.sin(hour_angle)
            if temp_now_in >= 30: heat_hours30_in += 1
            if temp_now_in >= 35: heat_hours35_in += 1

        monthly_heat30_base = heat_hours30_base * 30
        monthly_heat35_base = heat_hours35_base * 30
        monthly_heat30_in = heat_hours30_in * 30
        monthly_heat35_in = heat_hours35_in * 30

        # 產能
        t_diff = abs(t_in - crop['idealTemp'])
        score_temp = 1 - (t_diff / (crop['tempTolerance'] * 1.5))
        if t_in > 30:
            circ_bonus = 0
            if target_fan_specs['circCount'] > 0:
                covered_area = target_fan_specs['circCount'] * target_fan_specs['circDistance'] * 5
                if floor_area > 0:
                    coverage_ratio = min(1, covered_area / floor_area)
                    circ_bonus = coverage_ratio * 0.15
            score_temp *= (0.6 + circ_bonus)
        score_temp = max(0, min(1, score_temp))

        solar_in = solar_out * t_trans
        score_light = 1
        if solar_in < crop['lightSaturation']:
            if crop['lightSaturation'] > 0:
                deficit = (crop['lightSaturation'] - solar_in) / crop['lightSaturation']
                score_light = 1 - (deficit * crop['lightSlope'])
        score_light = max(0.1, min(1, score_light))

        efficiency = score_temp * score_light
        monthly_cycles = annual_cycles / 12
        monthly_yield = planting_area * planting_density * crop['baseWeight'] * efficiency * monthly_cycles
        price = market_prices[i]
        revenue = monthly_yield * price
        
        data.append({
            'month': i + 1, 'cropName': crop['name'],
            'tempOut': t_out, 'tempIn': t_in, 'rhIn': rh_in, 'vIn': v_in, 'ach': ach,
            'wind': wind_speed, 'solarIn': solar_in,
            'yield': monthly_yield, 'price': price, 'revenue': revenue,
            'efficiency': efficiency * 100,
            'heat30_Base': monthly_heat30_base, 'heat35_Base': monthly_heat35_base,
            'heat30_In': monthly_heat30_in, 'heat35_In': monthly_heat35_in,
        })
        
        total_yield += monthly_yield
        total_revenue += revenue

    return {
        'data': data, 'totalYield': total_yield, 'totalRevenue': total_revenue,
        'floorArea': floor_area, 'volume': volume, 'maxSummerTemp': max_summer_temp
    }

# ... (上面是你原本的 scan_and_load_weather_data 函式，不要動它) ...

# ==========================================
# 2. Google Sheets 資料庫連線 (新增在這下面)
# ==========================================
def load_google_sheet_db():
    """連線到 Google Sheets 讀取紀錄"""
    try:
        # 建立連線
        conn = st.connection("gsheets", type=GSheetsConnection)
        # 讀取資料 (假設你的工作表名稱叫做 'log_data'，若沒指定則讀第一張)
        df_db = conn.read(worksheet="工作表1") 
        return conn, df_db
    except Exception as e:
        st.error(f"無法連線到資料庫: {e}")
        return None, None

# ==========================================
# 3. 主程式邏輯 (Main App)
# ==========================================
# 載入氣象資料 (原本的功能)
weather_dict = scan_and_load_weather_data()

# 載入資料庫 (新的功能)
conn, df_db = load_google_sheet_db()

if df_db is not None:
    st.success("✅ 資料庫連線成功！")
    # 這裡可以開始寫你的 st.dataframe(df_db) 或 st.form...

# ==========================================
# 4. Streamlit UI 邏輯
# ==========================================

col_header_1, col_header_2 = st.columns([1, 4])
with col_header_1:
    st.image("https://cdn-icons-png.flaticon.com/512/2942/2942544.png", width=80) 
with col_header_2:
    st.title("溫室模擬與環境分析系統 V6.0")
    st.markdown("多地區氣候分析")

# --- 使用 Container 控制側邊欄順序 ---
settings_container = st.sidebar.container()

# 執行載入
imported_locations = scan_and_load_weather_data(base_folder='weather_data')
if imported_locations:
    LOCATION_DATABASE = imported_locations
else:
    LOCATION_DATABASE = BUILTIN_LOCATIONS

# 回填最上方位置
with settings_container:
    st.header("地區與基礎設定")
    location_options = list(LOCATION_DATABASE.keys())
    if not location_options:
        st.error("無可用地區資料")
        st.stop()
    location_id = st.selectbox("選擇模擬地區", location_options, format_func=lambda x: LOCATION_DATABASE[x]['name'])
    current_location = LOCATION_DATABASE[location_id]
st.info(current_location.get('description', '無描述資訊'))

# Session State 初始化
if 'monthly_crops' not in st.session_state:
    st.session_state.monthly_crops = ['lettuce'] * 12
if 'market_prices' not in st.session_state:
    st.session_state.market_prices = current_location['data']['marketPrice'].copy()
if 'planting_density' not in st.session_state:
    st.session_state.planting_density = 25.0
if 'annual_cycles' not in st.session_state:
    st.session_state.annual_cycles = 15.0

# 切換地區時重置價格
if 'last_location' not in st.session_state:
    st.session_state.last_location = location_id
if st.session_state.last_location != location_id:
    st.session_state.market_prices = current_location['data']['marketPrice'].copy()
    st.session_state.last_location = location_id

# 頁籤內容
tab1, tab2, tab3, tab4 = st.tabs([
    "1. 外部環境", "2. 內部微氣候", "3. 產能價格", "4. 邊際效益"
])

# --- Tab 1: 外部環境 ---
with tab1:
    st.subheader(f"📍 {current_location['name']} - 氣候數據")
    climate_data = current_location['data']
    
    df_climate = pd.DataFrame({
        'Month': climate_data['months'], 'Temp': climate_data['temps'],
        'MaxTemp': climate_data['maxTemps'], 'MinTemp': climate_data['minTemps'],
        'Humidity': climate_data['humidities'], 'Solar': climate_data['solar'],
        'Wind': climate_data['wind']
    })

    df_climate['Solar_W'] = df_climate['Solar'] * 11.574

    col1, col2 = st.columns(2)
    with col1:
        fig_temp = make_subplots(specs=[[{"secondary_y": True}]])
        fig_temp.add_trace(go.Bar(x=df_climate['Month'], y=df_climate['Temp'], name="平均氣溫", marker_color='orange', opacity=0.6), secondary_y=False)
        fig_temp.add_trace(go.Scatter(x=df_climate['Month'], y=df_climate['MaxTemp'], name="最高氣溫", line=dict(color='red', dash='dot')), secondary_y=False)
        fig_temp.add_trace(go.Scatter(x=df_climate['Month'], y=df_climate['MinTemp'], name="最低氣溫", line=dict(color='blue', dash='dot')), secondary_y=False)
        fig_temp.add_trace(go.Scatter(x=df_climate['Month'], y=df_climate['Solar_W'], name="平均日射 (W/m²)", line=dict(color='#f59e0b', width=3)), secondary_y=True)
        
        fig_temp.update_layout(
            title="月氣溫與日射量", 
            height=400, 
            template="plotly_white",
            # [修正] 強制 X 軸顯示每個月份
            xaxis=dict(
                tickmode='linear', # 線性刻度
                dtick=1,           # 每 1 單位顯示一個刻度
                tick0=1,           # 從 1 開始
                range=[0.5, 12.5], # 範圍稍微寬一點以免切到
                tickvals=list(range(1, 13)), 
                ticktext=[f"{i}月" for i in range(1, 13)]
            ),
            legend=dict(orientation="h", y=1.1)
        )
        fig_temp.update_yaxes(title_text="氣溫 (°C)", secondary_y=False)
        fig_temp.update_yaxes(title_text="日射強度 (W/m²)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_temp, use_container_width=True)

    with col2:
        scatter_points = []
        for i, m in enumerate(climate_data['months']):
            base_temp = climate_data['temps'][i]
            base_solar_w = df_climate['Solar_W'][i]
            for _ in range(30):
                sim_temp = base_temp + (np.random.random() - 0.5) * 6
                sim_solar = max(0, base_solar_w + (np.random.random() - 0.5) * 100)
                scatter_points.append({'Temp': min(40, max(0, sim_temp)), 'Solar_W': sim_solar})
                
        df_scatter = pd.DataFrame(scatter_points)
        fig_niche = px.scatter(df_scatter, x='Temp', y='Solar_W', opacity=0.3, title="氣候生態位 (光溫分佈)")
        first_row = df_climate.iloc[[0]]
        df_loop = pd.concat([df_climate, first_row], ignore_index=True)
        text_labels = [str(m)+"月" if i < 12 else "" for i, m in enumerate(df_loop['Month'])]
        fig_niche.add_trace(go.Scatter(x=df_loop['Temp'], y=df_loop['Solar_W'], mode='lines+markers+text', text=text_labels, textposition="top center", name='月均值', line=dict(color='#ea580c', width=3)))
        fig_niche.update_layout(xaxis_title="氣溫 (°C)", yaxis_title="日射強度 (W/m²)", height=400, template="plotly_white")
        st.plotly_chart(fig_niche, use_container_width=True)

# --- Tab 2: 內部微氣候 ---
with tab2:
    st.subheader("🏠 溫室結構與模擬")
    
    col_input, col_result = st.columns([1, 2])
    
    with col_input:
        with st.expander("1. 結構尺寸", expanded=True):
            gh_width = st.number_input("寬度 (m)", 25.0)
            gh_length = st.number_input("長度 (m)", 40.0)
            gh_height = st.number_input("簷高 (m)", 4.5)
            
            gh_roof = st.selectbox("屋頂形式", list(ROOF_OPTIONS.keys()), format_func=lambda x: ROOF_OPTIONS[x])
            
            if gh_roof == 'venlo': roof_angle = st.slider("屋頂斜度 (°)", 15, 30, 22)
            elif gh_roof == 'single slope': roof_angle = st.slider("屋頂斜度 (°)", 5, 45, 15)
            else: roof_angle = 0

            st.markdown("##### 🛡️ 覆蓋材料設定")
            mat_df = load_mat_database()
            if not mat_df.empty:
                mat_idx = st.selectbox("選擇覆蓋材料", mat_df.index, format_func=lambda x: mat_df['Label'][x])
                sel_mat = mat_df.loc[mat_idx]
                mat_code = str(sel_mat['Material_Code'])
                light_trans = float(sel_mat['Light_Transmittance_Rate'])
                is_thermic = str(sel_mat['Thermic'])
                mat_type = str(sel_mat['Material_Type'])
                
                if 'Glass' in mat_type: calc_u_val = 5.5
                elif is_thermic == 'Yes': calc_u_val = 4.5
                else: calc_u_val = 6.0
                
                MATERIAL_OPTIONS[mat_code] = {'label': mat_code, 'trans': light_trans, 'uValue': calc_u_val}
                gh_mat = mat_code
            else:
                gh_mat = st.selectbox("覆蓋材料", list(MATERIAL_OPTIONS.keys()), format_func=lambda x: MATERIAL_OPTIONS[x]['label'])

        with st.expander("2. 通風設備 (已連結設備庫)", expanded=True):
            elec_rate = st.number_input("⚡ 電費費率 ($/度)", value=4.0, step=0.5)
            st.session_state['elec_rate'] = elec_rate
            
            st.markdown("#### 💨 負壓排風扇 (Exhaust Fan)")
            ex_fans = load_fan_database(category_filter="Exhaust")
            if not ex_fans.empty:
                fan_idx = st.selectbox("選擇排風扇型號", ex_fans.index, format_func=lambda x: ex_fans['Label'][x])
                sel_fan = ex_fans.loc[fan_idx]
                fan_flow = float(sel_fan['Airflow_CMH'])
                fan_power = float(sel_fan['Power_W'])
                fan_price = float(sel_fan['Price_NTD'])
                st.info(f"📍 **{sel_fan['Model']}** | 風量: {int(fan_flow):,} CMH | 功率: {int(fan_power)} W")
                st.session_state['selected_fan_power'] = fan_power
                st.session_state['selected_fan_price'] = fan_price
            else:
                fan_flow = st.number_input("風量 (CMH)", 40000)
                fan_power = st.number_input("功率 (W)", 1000)
                st.session_state['selected_fan_power'] = fan_power

            fan_count = st.number_input("排風扇數量 (台)", 0)
            st.divider()
            
            st.markdown("#### 🔄 內部循環扇 (Circulation Fan)")
            circ_fans = load_fan_database(category_filter="Circulation")
            if circ_fans.empty: circ_fans = load_fan_database()
            if not circ_fans.empty:
                c_idx = st.selectbox("選擇循環扇型號", circ_fans.index, format_func=lambda x: circ_fans['Label'][x], key='circ_select')
                sel_circ = circ_fans.loc[c_idx]
                auto_dist = float(sel_circ['Throw_Distance_m']) if float(sel_circ['Throw_Distance_m']) > 0 else 15.0
            else: auto_dist = 15.0

            c1, c2 = st.columns(2)
            circ_dist = c1.number_input("循環扇吹距 (m)", value=auto_dist)
            circ_count = c2.number_input("循環扇數量 (台)", 0)

        with st.expander("3. 環控參數", expanded=True):
            shading = st.slider("遮蔭率 (%)", 0, 90, 30)
            st.markdown("##### 🕸️ 防蟲網設定")
            net_df = load_net_database()
            if not net_df.empty:
                net_idx = st.selectbox("選擇防蟲網規格", net_df.index, format_func=lambda x: net_df['Label'][x])
                sel_net = net_df.loc[net_idx]
                auto_openness = float(sel_net['Openness_Percent'])
                insect_net = st.number_input("實際通風率 (%)", value=auto_openness)
            else:
                insect_net = st.slider("防蟲網通風 (%)", 0, 100, 70)
            
            st.markdown("##### 🌱 栽培系統")
            cultivation_type = st.selectbox("選擇栽培模式", ["NFT (薄膜水耕)", "DFT (深水水耕)", "Soil (一般土耕)", "Pot (介質離地)"])
            vol_coef_map = {"NFT (薄膜水耕)": 1.1, "Pot (介質離地)": 1.2, "Soil (一般土耕)": 1.4, "DFT (深水水耕)": 1.6}
            auto_vol_coef = vol_coef_map[cultivation_type]

            roof_vent = st.number_input("天窗面積 (m²)", 0.0)
            side_vent = st.number_input("側窗面積 (m²)", 0.0)

        with st.expander("4. 噴霧降溫系統 (Fogging)", expanded=True):
                fog_df = load_fog_database()
                if not fog_df.empty:
                    fog_idx = st.selectbox("選擇噴霧量", fog_df.index, format_func=lambda x: f"{fog_df['Spray_Capacity_g_m2_hr'][x]} g/m²/hr")
                    sel_fog = fog_df.loc[fog_idx]
                    fog_cap = float(sel_fog['Spray_Capacity_g_m2_hr'])
                    
                    area_tmp = gh_width * gh_length
                    total_water_g_hr = fog_cap * area_tmp
                    cooling_power_w = (total_water_g_hr * 2450 / 3600) * 0.8
                    try:
                        est_vent_flow = (fan_count * fan_flow) / 3600 if fan_count > 0 else (area_tmp * 3) / 60
                    except: est_vent_flow = (area_tmp * 3) / 60
                    heat_removal_est = (est_vent_flow * 1200) + (6.0 * area_tmp * 1.5)
                    est_delta_t = cooling_power_w / heat_removal_est
                    
                    st.markdown("##### 🧪 物理推導效能")
                    c_f1, c_f2 = st.columns(2)
                    c_f1.metric("冷卻功率", f"{int(cooling_power_w/1000)} kW")
                    c_f2.metric("最大降溫潛力", f"-{est_delta_t:.1f} °C")
                    
                    st.markdown("##### ⚙️ 啟動邏輯")
                    fog_trigger_temp = st.slider("啟動溫度 (°C)", 25, 35, 28)
                    fog_stop_rh = st.slider("停止濕度 (%RH)", 70, 95, 85)
                else:
                    fog_cap = 0; fog_trigger_temp = 28; fog_stop_rh = 85

    # --- 物理係數運算與規格打包 ---
    floor_area = gh_width * gh_length
    rad = math.radians(roof_angle) if roof_angle > 0 else 0.5
    
    if gh_roof == 'tunnel':
        volume_coef = 1.15; surface_coef = 1.2; vent_efficiency = 0.8
    else:
        avg_roof_height = 0.5 * (gh_width if gh_roof == 'single slope' else 4.0) * math.tan(rad)
        volume_coef = 1 + (avg_roof_height / gh_height)
        surface_coef = 1 / math.cos(rad)
        vent_efficiency = 1.0 + (math.sin(rad) * 0.5)

    # 係數疊加修正
    vent_efficiency = vent_efficiency * (insect_net / 100.0) * 0.8
    volume_coef = volume_coef * auto_vol_coef

    gh_specs = {
        'width': gh_width, 'length': gh_length, 'gutterHeight': gh_height,
        'roofType': gh_roof, 'material': gh_mat, 
        'roofVentArea': roof_vent, 'sideVentArea': side_vent, 
        'shadingScreen': shading, 'insectNet': insect_net,
        '_vol_coef': volume_coef, '_surf_coef': surface_coef, '_vent_eff': vent_efficiency
    }
    
    fan_specs = {
        'exhaustCount': fan_count, 'exhaustFlow': fan_flow, 
        'circCount': circ_count, 'circDistance': circ_dist
    }

    # [重要] 存入 Session State 供 Tab 4 使用
    st.session_state.gh_specs = gh_specs
    st.session_state.fan_specs = fan_specs

    sim_results = run_simulation(
        gh_specs, fan_specs, current_location['data'], 
        st.session_state.monthly_crops, st.session_state.planting_density, 
        st.session_state.annual_cycles, st.session_state.market_prices
    )
    df_sim = pd.DataFrame(sim_results['data'])

    with col_result:
        st.markdown(f"""
        <div style="background-color:#000000; padding:10px; border-radius:8px; font-size:0.9em; border:1px solid #e2e8f0;">
            <b>📐 物理模型參數：</b> <br>
            • 表面積係數: <span style="color:blue">{surface_coef:.2f}</span> (散熱面積)<br>
            • 體積係數: <span style="color:green">{volume_coef:.2f}</span> (熱緩衝能力)<br>
            • 通風效率: <span style="color:orange">{vent_efficiency:.2f}</span> (結構與防蟲網影響)
        </div>
        """, unsafe_allow_html=True)
        
        fig_sim = make_subplots(specs=[[{"secondary_y": True}]])
        fig_sim.add_trace(go.Scatter(x=df_sim['month'], y=df_sim['tempOut'], fill='tozeroy', name="外溫", line=dict(color='#cbd5e1')), secondary_y=False)
        fig_sim.add_trace(go.Scatter(x=df_sim['month'], y=df_sim['tempIn'], name="室溫", line=dict(color='#ef4444', width=3)), secondary_y=False)
        fig_sim.add_trace(go.Bar(x=df_sim['month'], y=df_sim['vIn'], name="風速", marker_color='#2dd4bf', opacity=0.5), secondary_y=True)
        fig_sim.update_layout(title="微氣候模擬", height=350, template="plotly_white")
        st.plotly_chart(fig_sim, use_container_width=True)

        fig_heat = go.Figure()
        fig_heat.add_trace(go.Bar(x=df_sim['month'], y=df_sim['heat30_Base'], name='原況>30°C', marker_color='#9ca3af'))
        fig_heat.add_trace(go.Bar(x=df_sim['month'], y=df_sim['heat35_Base'], name='原況>35°C', marker_color='#ef4444'))
        fig_heat.add_trace(go.Bar(x=df_sim['month'], y=df_sim['heat30_In'], name='改善>30°C', marker_color='#86efac'))
        fig_heat.add_trace(go.Bar(x=df_sim['month'], y=df_sim['heat35_In'], name='改善>35°C', marker_color='#22c55e'))
        fig_heat.update_layout(title="高溫時數比較", barmode='group', height=300, template="plotly_white")
        st.plotly_chart(fig_heat, use_container_width=True)

    # ----------------------------------------------------------------
    # 24小時一日動態模擬
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("⏱️ 24小時一日動態模擬 (支援 CWA 時報表)")

    target_folder = 'weather_data' 
    current_sid = location_id 
    df_day_data = None 

    all_files = {}
    matched_files = {} 
    
    if os.path.exists(target_folder):
        files = [f for f in os.listdir(target_folder) if f.endswith('.csv')]
        files.sort(reverse=True)
        for f in files:
            full_path = os.path.join(target_folder, f)
            all_files[f] = full_path
            if current_sid in f: matched_files[f] = full_path

    col_h1, col_h2 = st.columns([1, 2])
    
    with col_h1:
        if matched_files:
            st.success(f"🎯 已鎖定測站檔案")
            file_options = list(matched_files.keys()); file_dict = matched_files
        else:
            if all_files:
                st.info(f"💡 顯示所有氣候檔")
                file_options = list(all_files.keys()); file_dict = all_files
            else:
                st.warning(f"⚠️ `{target_folder}` 資料夾是空的。"); file_options = []; file_dict = {}

        if file_options:
            sel_file = st.selectbox("1. 選擇氣候檔", file_options)
            csv_path = file_dict[sel_file]
            try:
                try: df_raw = pd.read_csv(csv_path, header=1, encoding='utf-8', on_bad_lines='skip')
                except: 
                    try: df_raw = pd.read_csv(csv_path, header=1, encoding='big5', on_bad_lines='skip')
                    except: df_raw = pd.read_csv(csv_path, header=0, encoding='utf-8', on_bad_lines='skip')

                df_raw.columns = [c.strip() for c in df_raw.columns]
                rmap = {}
                for c in df_raw.columns:
                    if '觀測時間' in c or 'Time' in c: rmap[c] = 'Time'
                    elif '氣溫' in c or 'Temp' in c: rmap[c] = 'Temp'
                    elif '全天空日射量' in c: rmap[c] = 'Solar'
                    elif '日射' in c and 'Solar' not in rmap.values(): rmap[c] = 'Solar'
                    elif '濕度' in c or 'RH' in c: rmap[c] = 'RH'
                    elif '平均風速' in c: rmap[c] = 'Wind'
                    elif '風速' in c and '瞬間' not in c and 'Wind' not in rmap.values(): rmap[c] = 'Wind'
                
                df_raw.rename(columns=rmap, inplace=True)
                df_raw['Time'] = pd.to_datetime(df_raw['Time'], errors='coerce')
                df_raw = df_raw.dropna(subset=['Time'])
                df_raw['DateStr'] = df_raw['Time'].dt.strftime('%Y-%m-%d')
                unique_dates = sorted(df_raw['DateStr'].unique(), reverse=True)
                target_date = st.selectbox("2. 選擇日期", unique_dates)
                
                mask = df_raw['DateStr'] == target_date
                df_day_data = df_raw[mask].copy().sort_values('Time')
                df_day_data['Hour'] = df_day_data['Time'].dt.hour
                
                if len(df_day_data) == 0: st.error(f"❌ 該日期無數據")
                else:
                    for col, def_val in [('Solar', 0.0), ('Wind', 1.0), ('RH', 75.0), ('Temp', 25.0)]:
                        if col not in df_day_data.columns: df_day_data[col] = def_val
                        else: df_day_data[col] = pd.to_numeric(df_day_data[col], errors='coerce').fillna(def_val)
                    t_avg = df_day_data['Temp'].mean(); s_sum = df_day_data['Solar'].sum()
                    st.info(f"📊 {target_date} 氣候摘要：\n• 均溫: {t_avg:.1f}°C\n• 總日射: {s_sum:.1f} MJ/m²")
            except Exception as e: st.error(f"檔案解析失敗: {e}")

    with col_h2:
        if df_day_data is not None and not df_day_data.empty:
            floor_area_h = gh_specs['width'] * gh_specs['length']
            vol_coef_h = gh_specs.get('_vol_coef', 1.2)
            surf_coef_h = gh_specs.get('_surf_coef', 1.15)
            vent_eff_h = gh_specs.get('_vent_eff', 1.0)
            wall_area_h = 2 * (gh_specs['width'] + gh_specs['length']) * gh_specs['gutterHeight']
            surface_area_h = (floor_area_h * surf_coef_h) + wall_area_h
            mat_props = MATERIAL_OPTIONS.get(gh_specs['material'], MATERIAL_OPTIONS['glass'])
            trans = mat_props['trans']
            u_val = mat_props['uValue']

            hourly_res = []
            for idx, row in df_day_data.iterrows():
                try:
                    t_out_h = float(row['Temp'])
                    solar_out_mj = float(row['Solar']) 
                    rh_out_h = float(row['RH'])
                    wind_h = float(row['Wind'])
                    hour_label = row['Hour']
                    
                    solar_out_w = solar_out_mj * 277.78
                    
                    t_trans_h = trans * (1 - gh_specs['shadingScreen']/100)
                    q_solar_h = (solar_out_mj * 1000000 / 3600) * floor_area_h * t_trans_h
                    
                    q_fog_h = 0; is_fogging = False
                    if fog_cap > 0 and t_out_h > (fog_trigger_temp - 2) and rh_out_h < fog_stop_rh:
                        total_water_g_hr = fog_cap * floor_area_h
                        q_fog_h = (total_water_g_hr * 2450 / 3600) * 0.8
                        is_fogging = True
                    
                    q_net_h = q_solar_h - q_fog_h
                    
                    vent_area_h = gh_specs['roofVentArea'] + gh_specs['sideVentArea']
                    nat_vent = wind_h * vent_area_h * 0.4 * (gh_specs['insectNet']/100) * vent_eff_h
                    force_vent = (fan_specs['exhaustCount'] * fan_specs['exhaustFlow']) / 3600
                    tot_vent = nat_vent + force_vent
                    
                    heat_removal = (tot_vent * 1200) + (u_val * surface_area_h)
                    delta_t_h = q_net_h / heat_removal if heat_removal > 0 else 0
                    t_in_h = t_out_h + delta_t_h
                    
                    if is_fogging and t_in_h < (t_out_h - 5): t_in_h = t_out_h - 5
                    
                    hourly_res.append({
                        'Time': hour_label, 'TempOut': t_out_h, 'TempIn': t_in_h, 'Solar_W': solar_out_w, 'Fog_On': 1 if is_fogging else 0
                    })
                except: continue

            if hourly_res:
                df_res_24 = pd.DataFrame(hourly_res)
                fig_24 = make_subplots(specs=[[{"secondary_y": True}]])
                fig_24.add_trace(go.Scatter(x=df_res_24['Time'], y=df_res_24['TempOut'], name="外氣溫", line=dict(color='#94a3b8', dash='dot')), secondary_y=False)
                fig_24.add_trace(go.Scatter(x=df_res_24['Time'], y=df_res_24['TempIn'], name="室內溫", mode='lines', line=dict(color='#dc2626', width=3)), secondary_y=False)
                fig_24.add_trace(go.Scatter(x=df_res_24['Time'], y=df_res_24['Solar_W'], name="日射強度 (W/m²)", fill='tozeroy', line=dict(color='#fbbf24', width=0), opacity=0.3), secondary_y=True)
                fig_24.update_layout(
                    title=f" {target_date} 24小時模擬 ({current_sid})", 
                    height=350, 
                    hovermode="x unified",
                    template="plotly_white",
                    # [修正] 強制 X 軸範圍為 0 到 24
                    xaxis=dict(
                        title="時間 (小時)", 
                        tickmode='linear', 
                        dtick=2, # 每 2 小時顯示一個刻度
                        range=[0, 24] # 強制鎖定範圍
                    ),
                    legend=dict(orientation="h", y=1.1)
                )
                fig_24.update_yaxes(title_text="溫度 (°C)", secondary_y=False)
                fig_24.update_yaxes(title_text="日射強度 (W/m²)", secondary_y=True, showgrid=False)
                st.plotly_chart(fig_24, use_container_width=True)
                
                mx = df_res_24['TempIn'].max(); dif = mx - df_res_24['TempIn'].min()
                c1, c2 = st.columns(2)
                c1.metric("最高室溫", f"{mx:.1f}°C"); c2.metric("日夜溫差", f"{dif:.1f}°C")

# --- Tab 3: 經濟分析 ---
with tab3:
    st.subheader("💰 經濟分析與價格管理")
    PRICE_DB = scan_and_load_market_prices(base_folder='market_data')
    if PRICE_DB: st.success(f"✅ 已連結 {len(PRICE_DB)} 檔市場價格")
    else: st.warning("⚠️ market_data 為空")

    c1, c2 = st.columns([1, 2])
    with c1:
        with st.form("econ_form"):
            st.markdown("#### 生產參數")
            den = st.number_input("種植密度 (株/m²)", value=st.session_state.planting_density)
            cyc = st.number_input("年周轉率 (次/年)", value=st.session_state.annual_cycles)
            
            c_names = [v['name'] for v in CROP_DATABASE.values()]
            curr_c = [CROP_DATABASE[i]['name'] for i in st.session_state.monthly_crops]
            
            dedit = st.data_editor(
                pd.DataFrame({'M': range(1, 13), 'C': curr_c, 'P': st.session_state.market_prices}),
                column_config={"M": st.column_config.NumberColumn("月", disabled=True), "C": st.column_config.SelectboxColumn("作物", options=c_names), "P": st.column_config.NumberColumn("批發價 ($)", min_value=0)},
                hide_index=True, use_container_width=True, height=300
            )
            auto_fill = st.checkbox("🔄 自動帶入 CSV 價格", value=True)
            sub = st.form_submit_button("🚀 計算", type="primary")
            
        if sub:
            st.session_state.planting_density = den
            st.session_state.annual_cycles = cyc
            n_map = {v['name']: k for k, v in CROP_DATABASE.items()}
            new_crops = []; new_prices = []
            for idx, row in dedit.iterrows():
                crop_name = row['C']; manual_price = row['P']
                new_crops.append(n_map[crop_name])
                matched_price = None
                if auto_fill and PRICE_DB:
                    for db_name in PRICE_DB.keys():
                        if crop_name in db_name: matched_price = PRICE_DB[db_name][idx]; break
                new_prices.append(matched_price if matched_price is not None else manual_price)
            st.session_state.monthly_crops = new_crops
            st.session_state.market_prices = new_prices
            st.rerun()

    with c2:
        sim_res = run_simulation(gh_specs, fan_specs, current_location['data'], st.session_state.monthly_crops, st.session_state.planting_density, st.session_state.annual_cycles, st.session_state.market_prices)
        df_res = pd.DataFrame(sim_res['data'])
        k1, k2, k3 = st.columns(3)
        k1.metric("預估年營收", f"${int(sim_res['totalRevenue']):,}")
        k2.metric("預估年產量", f"{sim_res['totalYield']/1000:.1f} 噸")
        avg_ef = df_res['efficiency'].mean()
        k3.metric("平均環境效率", f"{avg_ef:.1f}%")
        
        fig_ec = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ec.add_trace(go.Bar(x=df_res['month'], y=df_res['revenue'], name="營收", marker_color='#10b981', opacity=0.7), secondary_y=False)
        fig_ec.add_trace(go.Scatter(x=df_res['month'], y=df_res['yield'], name="產量", line=dict(color='blue', width=3)), secondary_y=True)
        fig_ec.update_layout(height=350, template="plotly_white", title="營收產量趨勢", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_ec, use_container_width=True)

# --- Tab 4: 邊際效益 ---
with tab4:
    st.subheader("⚖️ 邊際效益分析：產能與運行成本的最佳平衡")
    
    # [防呆] 確保規格已讀取
    if 'gh_specs' not in st.session_state:
        st.warning("⚠️ 請先至「Tab 2: 內部微氣候」設定溫室規格後，再進行效益分析。")
        st.stop()
    else:
        gh_specs = st.session_state.gh_specs
        fan_specs = st.session_state.fan_specs

    col_m1, col_m2 = st.columns([1, 2])
    
    with col_m1:
        st.markdown("### 1️⃣ 設定分析目標")
        m_var = st.selectbox("分析變數 (X軸)", ['exhaustCount', 'roofVent', 'sideVent', 'shadingScreen'], 
                             format_func=lambda x: {'exhaustCount':'負壓扇數量', 'roofVent':'天窗面積', 'sideVent':'側窗面積', 'shadingScreen':'遮蔭率'}[x])
        
        p_rate = st.session_state.get('elec_rate', 4.0)
        unit_power = st.session_state.get('selected_fan_power', 1000.0)
        
        st.markdown("### 2️⃣ 成本參數設定")
        with st.container(border=True):
            if m_var == 'exhaustCount':
                st.markdown("#### 🕒 風扇運轉設定")
                run_hours = st.number_input("年運轉時數 (hr)", value=4000, step=100)
                roof_unit_cost = 0 
            elif m_var == 'roofVent':
                st.markdown("#### 🏗️ 天窗成本設定")
                roof_unit_cost = st.number_input("天窗每 m² 造價/年攤提 ($)", value=200.0, step=50.0)
                run_hours = 4000 
            else:
                st.info("此項目目前僅分析產能變化。")
                run_hours = 4000; roof_unit_cost = 0

        with st.container(border=True):
            st.markdown("#### 🔒 背景固定條件")
            if m_var != 'exhaustCount': st.write(f"• 固定風扇: {fan_specs['exhaustCount']} 台")
            if m_var != 'roofVent': st.write(f"• 固定天窗: {gh_specs['roofVentArea']} m²")
            st.write(f"• 電費費率: ${p_rate}/度")

    fix_gh = gh_specs.copy()
    fix_fan = fan_specs.copy()
    
    if m_var == 'exhaustCount': x_start, x_end, x_step = 0, 5000, 1
    elif m_var == 'shadingScreen': x_start, x_end, x_step = 0, 90, 1
    elif m_var == 'sideVent':
        perimeter = 2 * (gh_specs['width'] + gh_specs['length'])
        max_h = min(4.0, gh_specs['gutterHeight'])
        max_side_area = int(perimeter * max_h)
        x_start, x_end, x_step = 0, max_side_area, 10
    elif m_var == 'roofVent':
        floor_area_tmp = gh_specs['width'] * gh_specs['length']
        max_roof_area = int(floor_area_tmp * 0.4)
        x_start, x_end, x_step = 0, max_roof_area, 10
    
    x_values = range(x_start, x_end + 1, x_step)
    m_pts = []  
    
    for v in x_values:
        i_gh = fix_gh.copy(); i_fan = fix_fan.copy()
        if m_var == 'exhaustCount': i_fan['exhaustCount'] = v
        elif m_var == 'roofVent': i_gh['roofVentArea'] = v
        elif m_var == 'sideVent': i_gh['sideVentArea'] = v
        elif m_var == 'shadingScreen': i_gh['shadingScreen'] = v
        
        r = run_simulation(i_gh, i_fan, current_location['data'], st.session_state.monthly_crops, st.session_state.planting_density, st.session_state.annual_cycles, st.session_state.market_prices)
        
        total_cost = 0
        curr_fan_cnt = v if m_var == 'exhaustCount' else fix_fan['exhaustCount']
        elec_cost = curr_fan_cnt * (unit_power / 1000) * run_hours * p_rate
        total_cost += elec_cost
        if m_var == 'roofVent': total_cost += (v * roof_unit_cost)
            
        net_profit = r['totalRevenue'] - total_cost
        m_pts.append({'變數值': v, '年產量 (kg)': int(r['totalYield']), '年產值 ($)': int(r['totalRevenue']), '總成本 ($)': int(total_cost), '淨利 ($)': int(net_profit)})
    
    df_m = pd.DataFrame(m_pts)
    df_m['產值增量'] = df_m['年產值 ($)'].diff().fillna(0)
    df_m['成本增量'] = df_m['總成本 ($)'].diff().fillna(0)
    df_m['邊際效益(ROI)'] = df_m.apply(lambda x: x['產值增量']/x['成本增量'] if x['成本增量']>0 else 0, axis=1)

    with col_m2:
        fig_m = make_subplots(specs=[[{"secondary_y": True}]])
        fig_m.add_trace(go.Scatter(x=df_m['變數值'], y=df_m['年產量 (kg)'], name="作物年產量 (kg)", mode='lines+markers', line=dict(color='#3b82f6', width=3, dash='dot'), marker=dict(size=6)), secondary_y=True)
        fig_m.add_trace(go.Scatter(x=df_m['變數值'], y=df_m['淨利 ($)'], name="扣除電費後淨利 ($)", mode='lines', fill='tozeroy', line=dict(color='#15803d', width=2), opacity=0.1), secondary_y=False)
        fig_m.add_trace(go.Scatter(x=df_m['變數值'], y=df_m['總成本 ($)'], name="總成本 ($)", mode='lines', line=dict(color='#ef4444', width=3)), secondary_y=False)
        
        x_label = {'exhaustCount':'風扇數量 (台)', 'roofVent':'天窗面積 (m²)', 'sideVent':'側窗面積 (m²)', 'shadingScreen':'遮蔭率 (%)'}[m_var]
        fig_m.update_layout(title=f"📊 {x_label} 最佳化分析", xaxis_title=x_label, hovermode="x unified", template="plotly_white", legend=dict(orientation="h", y=1.1))
        fig_m.update_yaxes(title_text="金額 ($)", secondary_y=False); fig_m.update_yaxes(title_text="產量 (kg)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_m, use_container_width=True)
        
        best_idx = df_m['淨利 ($)'].idxmax()
        best_x = df_m.loc[best_idx, '變數值']; best_yield = df_m.loc[best_idx, '年產量 (kg)']
        diminish_points = df_m[(df_m['邊際效益(ROI)'] < 1.0) & (df_m['邊際效益(ROI)'] > 0)]
        warning_x = diminish_points['變數值'].min() if not diminish_points.empty else None

        st.markdown("#### 💡 決策建議")
        c_res1, c_res2 = st.columns(2)
        c_res1.success(f"**🏆 最佳獲利點**\n 當 **{x_label} = {int(best_x)}** 時：\n• 年產量：**{int(best_yield):,} kg**")
        
        if warning_x and warning_x > best_x: c_res2.warning(f"**⚠️ 邊際效益遞減**\n 當 **{x_label} 超過 {int(warning_x)}** 時：\n每多花 $1 成本，增加產值 < $1。")
        elif warning_x: c_res2.info(f"注意：超過 **{int(warning_x)}** 後，效益開始下降。")
        else: c_res2.info("此範圍內增加投入均為正向收益。")
             
        with st.expander("查看詳細數據表 (含 ROI 分析)"):

            st.dataframe(df_m[['變數值', '年產量 (kg)', '總成本 ($)', '淨利 ($)', '邊際效益(ROI)']].style.format("{:,.0f}"), use_container_width=True)
