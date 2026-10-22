import math
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# =========================
# CONFIG (à changer ici)
# =========================
INPUT_FILE = r"~/PycharmProjects/Panneaux/Parcours.xlsx"
SHEET_NAME = "Panneaux permanents"
PROXY = "http://vip-users.proxy.edf.fr:3131"

SERVICE_SECONDS = 5 * 60                # 5 min par arrêt
TARGET_SEGMENT_SECONDS = 90 * 60        # 1h30

# Choix du mode de découpage :
# - "TARGET_DURATION" : segments ~ 1h30 (souvent tailles inégales)
# - "EQUAL_COUNT"     : N segments de durée équivalente (évite segments trop petits)
SPLIT_MODE = "EQUAL_COUNT"

# En mode EQUAL_COUNT :
#   - mets un entier (ex: 8) pour imposer N
#   - ou laisse None pour auto-calculer N ≈ total / 1h30
NUM_SEGMENTS = None

# Google Maps : 10 lieux max par lien (origine comprise) => 9 stops après l'origine
MAX_LOCS_PER_GOOGLE_LINK = 10

# Remplace None d'OSRM par un coût énorme
BIG = 10**9

# =========================
# OUT DIR (Bureau)
# =========================
def get_desktop_dir() -> Path:
    home = Path.home()
    for name in ["Desktop", "Bureau"]:
        p = home / name
        if p.exists():
            return p
    return Path.cwd()

OUT_DIR = get_desktop_dir()

# =========================
# HTTP session + retry + proxy
# =========================
SESSION = requests.Session()
SESSION.trust_env = False
SESSION.proxies.update({"http": PROXY, "https": PROXY})

retry = Retry(
    total=8,
    backoff_factor=0.8,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)
adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
SESSION.mount("https://", adapter)
SESSION.mount("http://", adapter)

# =========================
# Lecture fichier (csv/xlsx) + encodages Windows
# =========================
def read_points(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=SHEET_NAME)

    encodings = ["utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except Exception as e:
            last_err = e
    raise last_err

def to_float(v):
    if pd.isna(v):
        return math.nan
    s = str(v).strip().replace(",", ".")
    try:
        return float(s)
    except:
        return math.nan

# =========================
# Géocodage GeoPF (seulement pour lat/lon manquants)
# =========================
def geocode_geopf(q: str, limit: int = 1):
    url = "https://data.geopf.fr/geocodage/search"
    params = {"q": q, "limit": limit}
    r = SESSION.get(url, params=params, timeout=(10, 30))
    if r.status_code != 200:
        print(f"[GEOCODE] HTTP {r.status_code} for: {q}")
        return None
    data = r.json()
    feats = data.get("features", [])
    if not feats:
        return None
    lon, lat = feats[0]["geometry"]["coordinates"]
    return float(lat), float(lon)

def ensure_latlon(df):
    missing = df["lat"].isna() | df["lon"].isna()
    print("Points sans coordonnées à géocoder:", int(missing.sum()), "sur", len(df))

    for i, row in df[missing].iterrows():
        q = str(row.get("query_geocode", "")).strip()
        ll = geocode_geopf(q)
        if ll is None:
            print(f"[GEOCODE] pas trouvé: {row.get('number')} | {q}")
        else:
            df.at[i, "lat"] = float(ll[0])
            df.at[i, "lon"] = float(ll[1])
        time.sleep(0.2)

    return df

# =========================
# OSRM table (durations)
# =========================
def osrm_table_seconds(latlons):
    coords = ";".join([f"{lon},{lat}" for lat, lon in latlons])
    url = f"https://router.project-osrm.org/table/v1/driving/{coords}"
    r = SESSION.get(url, params={"annotations": "duration"}, timeout=60)
    r.raise_for_status()
    data = r.json()
    dur = data["durations"]
    none_count = sum(1 for row in dur for x in row if x is None)
    if none_count:
        print(f"[WARN] OSRM a renvoyé {none_count} durées None -> BIG")
    return [[int(x) if x is not None else BIG for x in row] for row in dur]

# =========================
# Open-TSP (départ & arrivée inconnus) via 2 noeuds fictifs
# =========================
def add_dummy_start_end(dur_real):
    n = len(dur_real)
    ds = n
    de = n + 1
    m = n + 2
    d2 = [[BIG] * m for _ in range(m)]
    for i in range(n):
        for j in range(n):
            d2[i][j] = int(dur_real[i][j])
    for j in range(n):
        d2[ds][j] = 0
    for i in range(n):
        d2[i][de] = 0
    for i in range(m):
        d2[i][ds] = BIG
        d2[de][i] = BIG
    d2[ds][de] = 0
    d2[ds][ds] = 0
    d2[de][de] = 0
    return d2, ds, de

def solve_fixed_ends(dur, start_node, end_node, time_limit_s=45):
    n = len(dur)
    manager = pywrapcp.RoutingIndexManager(n, 1, [start_node], [end_node])
    routing = pywrapcp.RoutingModel(manager)

    def cb(from_i, to_i):
        a = manager.IndexToNode(from_i)
        b = manager.IndexToNode(to_i)
        return dur[a][b]

    transit_idx = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(time_limit_s)

    sol = routing.SolveWithParameters(params)
    if sol is None:
        raise RuntimeError("Pas de solution trouvée (matrice incomplète ?).")

    order = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        order.append(manager.IndexToNode(idx))
        idx = sol.Value(routing.NextVar(idx))
    order.append(manager.IndexToNode(idx))
    return order

def solve_open_tsp_unknown_ends(dur_real, time_limit_s=45):
    d2, ds, de = add_dummy_start_end(dur_real)
    order2 = solve_fixed_ends(d2, ds, de, time_limit_s=time_limit_s)
    return [i for i in order2 if i not in (ds, de)]

# =========================
# Temps cumulés sur l'ordre
# delta(node) = travel(prev->node) + SERVICE
# =========================
def compute_deltas_and_cum(order, durations):
    deltas = []
    cum = []
    total = 0
    prev = None
    for node in order:
        travel = 0 if prev is None else durations[prev][node]
        delta = travel + SERVICE_SECONDS
        deltas.append(delta)
        total += delta
        cum.append(total)
        prev = node
    return deltas, cum, total

# =========================
# Split modes
# =========================
def split_target_duration(order, deltas, target_seconds):
    segments = []
    cur = []
    cur_t = 0
    for node, delta in zip(order, deltas):
        if cur and (cur_t + delta > target_seconds):
            segments.append(cur)
            cur = []
            cur_t = 0
        cur.append(node)
        cur_t += delta
    if cur:
        segments.append(cur)
    return segments

def split_equal_count(order, cum_times, n_segments):
    # frontières idéales : k * total / n
    total = cum_times[-1]
    targets = [k * total / n_segments for k in range(1, n_segments)]
    cut_indices = []
    j = 0
    for t in targets:
        while j < len(cum_times) and cum_times[j] < t:
            j += 1
        cut_indices.append(j)

    # rendre les cuts strictement croissants et valides
    fixed = []
    last = -1
    for idx in cut_indices:
        idx = max(idx, last + 1)
        idx = min(idx, len(order) - 2)  # pas couper après le dernier
        fixed.append(idx)
        last = idx

    segments = []
    start = 0
    for cut in fixed:
        segments.append(order[start:cut+1])
        start = cut+1
    segments.append(order[start:])
    return segments

# =========================
# Google Maps links (segment -> 1..N liens)
# =========================
def google_maps_link(origin_node, stops_nodes, df):
    def ll(i): return f"{df.loc[i,'lat']},{df.loc[i,'lon']}"
    if not stops_nodes:
        return None
    origin = ll(origin_node)
    destination = ll(stops_nodes[-1])
    waypoints = "|".join(ll(i) for i in stops_nodes[:-1])
    base = "https://www.google.com/maps/dir/?api=1"
    url = f"{base}&origin={urllib.parse.quote(origin)}&destination={urllib.parse.quote(destination)}&travelmode=driving"
    if waypoints:
        url += f"&waypoints={urllib.parse.quote(waypoints)}"
    return url

def links_for_segment(origin_node, segment_nodes, df):
    max_stops = MAX_LOCS_PER_GOOGLE_LINK - 1  # 9
    stops = segment_nodes[:]
    origin = origin_node
    links = []
    while stops:
        chunk = stops[:max_stops]
        links.append(google_maps_link(origin, chunk, df))
        origin = chunk[-1]
        stops = stops[max_stops:]
    return links

# =========================
# MAIN
# =========================
df = read_points(INPUT_FILE)

# Si ta colonne s'appelle "adress" dans Excel
df = df.rename(columns={"adress": "address"})

# Sauver l'ordre des colonnes d'origine (avant d'ajouter des colonnes techniques)
ORIGINAL_COLS = df.columns.tolist()

# Colonnes minimales (number = id du point)
for col in ["number", "name", "address", "city"]:
    if col not in df.columns:
        df[col] = ""

# lat/lon optionnels dans le fichier d'entrée
if "lat" not in df.columns:
    df["lat"] = math.nan
if "lon" not in df.columns:
    df["lon"] = math.nan

def to_float(v):
    if pd.isna(v):
        return math.nan
    s = str(v).strip().replace(",", ".")
    try:
        return float(s)
    except:
        return math.nan

df["lat"] = df["lat"].apply(to_float)
df["lon"] = df["lon"].apply(to_float)

# query géocodage (servira uniquement si lat/lon manquent)
df["query_geocode"] = (
    df["address"].fillna("").astype(str).str.strip()
    + ", "
    + df["city"].fillna("").astype(str).str.strip()
    + ", France"
).str.replace(r"\s+", " ", regex=True).str.strip()

df = ensure_latlon(df)

# forçage manuel si besoin
num = df["number"].astype(str).str.strip()
for k, (lat, lon) in FORCED_COORDS.items():
    mask = (num == str(k))
    if mask.any():
        df.loc[mask, "lat"] = float(lat)
        df.loc[mask, "lon"] = float(lon)
        print(f"[OK] Point {k} forcé -> ({lat}, {lon})")

# stop si encore des coordonnées manquantes
missing = df["lat"].isna() | df["lon"].isna()
if missing.any():
    bad = df.loc[missing, ["number", "name", "address", "city", "query_geocode"]]
    bad_path = OUT_DIR / "points_a_corriger.csv"
    bad.to_csv(bad_path, index=False, encoding="utf-8-sig")
    raise RuntimeError(f"Il reste {missing.sum()} points sans coordonnées. Liste -> {bad_path}")

# sauvegarde “enrichie” réutilisable comme entrée
enriched_path = OUT_DIR / "Parcours_avec_coordonnees.csv"
df[["number", "name", "address", "city", "lat", "lon"]].to_csv(enriched_path, index=False, encoding="utf-8-sig")
print("OK ->", enriched_path)

# matrice OSRM
latlons = list(zip(df["lat"].tolist(), df["lon"].tolist()))
dur = osrm_table_seconds(latlons)

# ordre optimisé (trajet linéaire, extrémités libres)
order = solve_open_tsp_unknown_ends(dur)

# temps & découpage
deltas, cum, total = compute_deltas_and_cum(order, dur)
total_h = total / 3600
print(f"Temps total estimé (trajet+pauses): {total_h:.2f} h")

if SPLIT_MODE == "TARGET_DURATION":
    segments = split_target_duration(order, deltas, TARGET_SEGMENT_SECONDS)
    print("Segments (TARGET_DURATION):", len(segments))
    print("=> pour EQUAL_COUNT, mets NUM_SEGMENTS =", len(segments))
elif SPLIT_MODE == "EQUAL_COUNT":
    nseg = NUM_SEGMENTS
    if nseg is None:
        nseg = math.ceil(total / TARGET_SEGMENT_SECONDS)
        print("NUM_SEGMENTS auto =", nseg)
    segments = split_equal_count(order, cum, nseg)
    print("Segments (EQUAL_COUNT):", len(segments))
else:
    raise ValueError("SPLIT_MODE doit être 'TARGET_DURATION' ou 'EQUAL_COUNT'")

# export CSV (1 ligne par point)
rows = []
for s_idx, seg in enumerate(segments, start=1):
    for j, node in enumerate(seg):
        rows.append({
            "segment": s_idx,
            "seq_in_segment": j,
            "number": df.loc[node, "number"],
            "name": df.loc[node, "name"],
            "address": df.loc[node, "address"],
            "city": df.loc[node, "city"],
            "lat": df.loc[node, "lat"],
            "lon": df.loc[node, "lon"],
        })

route_csv = OUT_DIR / "route_segments.csv"
pd.DataFrame(rows).to_csv(route_csv, index=False, encoding="utf-8-sig")
print("OK ->", route_csv, "| lignes:", len(rows))

# export liens Google Maps (plusieurs liens par segment si besoin)
links_txt = OUT_DIR / "google_maps_links.txt"
prev_end = None
with open(links_txt, "w", encoding="utf-8") as f:
    for s_idx, seg in enumerate(segments, start=1):
        # Toujours démarrer au 1er point du segment
        origin = seg[0]
        seg_nodes = seg[1:]   # stops après l'origine

        links = links_for_segment(origin, seg_nodes, df)

        if not links:
            f.write(f"Segment {s_idx}: 1 point (pas de lien)\n")
        else:
            for i, link in enumerate(links, start=1):
                f.write(f"Segment {s_idx}.{i}: {link}\n")

print("OK ->", links_txt)
# mapping node -> segment/seq
node_to_seg = {}
node_to_seq = {}
for s_idx, seg in enumerate(segments, start=1):
    for j, node in enumerate(seg):
        node_to_seg[node] = s_idx
        node_to_seq[node] = j

# DF route dans l'ordre optimisé
df_route = df.iloc[order].copy()
df_route.insert(0, "segment", [node_to_seg[i] for i in order])
df_route.insert(1, "seq_in_segment", [node_to_seq[i] for i in order])

# Reconstituer les colonnes : (segment, seq) + colonnes originales
cols_out = ["segment", "seq_in_segment"] + [c for c in ORIGINAL_COLS if c in df_route.columns]

# Si lat/lon n'étaient pas dans les colonnes d'origine, on peut quand même les garder à la fin
for c in ["lat", "lon"]:
    if c in df_route.columns and c not in cols_out:
        cols_out.append(c)

df_out = df_route[cols_out]

# sortie sur le Bureau
out_xlsx = OUT_DIR / "Parcours_segmenté.xlsx"
df_out.to_excel(out_xlsx, index=False, sheet_name=SHEET_NAME)
print("OK ->", out_xlsx)
print("OK ->", links_txt)