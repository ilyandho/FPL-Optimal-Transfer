import requests
import json
import time
import pandas as pd
import os
import concurrent.futures
import threading

# 071 401 2550
# poniZION1

# --- 1. CONFIGURATION & AUTOMATED LOGIN ---
raw_username = input("Please paste your username: ")
raw_password = input("Please paste your password: ")

USERNAME = raw_username.strip()
PASSWORD = raw_password.strip()

base_url = "https://api.fantasygo.io/public"
cdn_url = "https://cloudcdn.fantasygo.io/contest-event-team-players/contest_event_team_players_{}.json"

# Threading tools
lock = threading.Lock()
session = requests.Session()

def get_automated_token(username, password):
    """Logs into FantasyGo and retrieves a fresh Bearer token."""
    print(f"Attempting to log in with username: {username}...")
    login_query = """
    mutation Login($args: LoginInput!) {
      login(args: $args) {
        token
        refreshToken
        __typename
      }
    }
    """
    login_vars = {"args": {"mobile": username, "password": password}}

    try:
        response = session.post(base_url, json={
            "operationName": "Login",
            "query": login_query,
            "variables": login_vars
        })
        data = response.json()
        if "errors" in data:
            print(f"Login failed: {data['errors'][0].get('message')}")
            return None
        token = data['data']['login']['token']
        print("Login successful! Token retrieved.")
        return token
    except Exception as e:
        print(f"Error during login: {e}")
        return None

# Authenticate
auth_token = get_automated_token(USERNAME, PASSWORD)
if not auth_token:
    raw_token = input("Fallback: Please paste your Bearer Token manually: ")
    auth_token = raw_token.replace("Bearer ", "").strip()

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {auth_token}",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
    "Origin": "https://app.fantasygo.io",
    "Referer": "https://app.fantasygo.io/"
}
session.headers.update(headers)

# File paths
LEADERBOARD_CACHE = "raw_leaderboards.json"
PROGRESS_FILE = "processed_results.json"

# --- 2. STAGE 1: DISCOVERY ---
if not os.path.exists(LEADERBOARD_CACHE):
    print("--- STAGE 1: Discovery Phase ---")
    ended_contests_query = """
    query EndedContests($args: EndedContestsInput) {
      ended_contests: endedContests(args: $args) {
        contests {
          id
          name
          isSingleMatchContest
          entryFees
          userContestCount
          completedAt
          event { name season }
        }
      }
    }
    """
    vars_contests = {"args": {"eventTypes": ["englishPremierLeague"], "gameTypes": ["fantasy"], "limit": 3000, "offset": 0}}

    try:
        res = session.post(base_url, json={"operationName": "EndedContests", "query": ended_contests_query, "variables": vars_contests}).json()
        all_contests = res['data']['ended_contests']['contests']
        gameweeks = [c for c in all_contests if not c['isSingleMatchContest']]
    except Exception as e:
        print(f"Critical Connection Error: {e}")
        gameweeks = []

    leaderboard_data = []
    for gw in gameweeks:
        print(f"  Fetching leaderboard for: {gw['name']}")
        lb_query = """
        query LeaderboardForContest($args: LeaderboardForContestInput!) {
          leaderboard: leaderboardForContest(args: $args) {
            items: contestLeaderboardItems { userContestId username rank totalPoints }
          }
        }
        """
        lb_vars = {"args": {"contestId": int(gw['id']), "limit": 3000, "offset": 0, "entryFee": 10}}
        lb_res = session.post(base_url, json={"operationName": "LeaderboardForContest", "query": lb_query, "variables": lb_vars}).json()
        entries = lb_res.get('data', {}).get('leaderboard', {}).get('items', [])

        for entry in entries:
            entry.update({
                'gameweek_name': gw['name'],
                'gameweek_id': gw['id'],
                'entryFees': ", ".join(map(str, gw.get('entryFees', []))),
                'userContestCount': gw.get('userContestCount'),
                'completedAt': gw.get('completedAt'),
                'event_name': gw.get('event', {}).get('name'),
                'event_season': gw.get('event', {}).get('season')
            })
            leaderboard_data.append(entry)
        time.sleep(0.1)

    with open(LEADERBOARD_CACHE, 'w') as f:
        json.dump(leaderboard_data, f, indent=4)

# --- 3. STAGE 2: PARALLEL EXTRACTION ---
print("\n--- STAGE 2: Parallel Extraction Phase ---")

processed_data = []
if os.path.exists(PROGRESS_FILE):
    try:
        with open(PROGRESS_FILE, 'r') as f:
            processed_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: {PROGRESS_FILE} is corrupted. Attempting to load from backup or starting fresh.")
        # Optional: Add logic to check for a .tmp or .bak file here
        processed_data = []

processed_ids = {item['userContestId'] for item in processed_data}
with open(LEADERBOARD_CACHE, 'r') as f:
    master_list = json.load(f)

to_process = [item for item in master_list if item['userContestId'] not in processed_ids]
player_mappings = {}

def get_stat(stats_list, stat_type):
    if not stats_list: return "0"
    for s in stats_list:
        if s.get('type') == stat_type: return s.get('value')
    return "0"

def save_progress_atomically(data):
    """Writes to a temporary file then renames it to prevent corruption."""
    temp_file = PROGRESS_FILE + ".tmp"
    with open(temp_file, 'w') as f:
        json.dump(data, f, indent=4)
    os.replace(temp_file, PROGRESS_FILE)

def process_entry(entry):
    global auth_token
    cid = str(entry['gameweek_id'])

    # 1. Fetch Player Mapping if not cached (Thread Safe)
    with lock:
        if cid not in player_mappings:
            try:
                map_res = requests.get(cdn_url.format(cid), timeout=10)
                if map_res.status_code == 200:
                    raw_players = map_res.json()
                    player_mappings[cid] = {
                        str(p['id']): {
                                "displayName": p.get('displayName'),
                                "firstName": p.get('firstName'),
                                "lastName": p.get('lastName'),
                                "position": p.get('position'),
                                "cost": p.get('cost'),
                                "team": p.get('team', {}).get('name'),
                                "goals": get_stat(p.get('statistics', []), "goalsScored"),
                                "assists": get_stat(p.get('statistics', []), "assists")
                        } for p in raw_players
                    }
                else: player_mappings[cid] = {}
            except: player_mappings[cid] = {}

    # 2. Fetch User Team
    team_query = """
    query UserContest($args: GetUserContestInput!) {
      user_contest: userContest(args: $args) {
        team: userContestTeam {
          formation
          players: userContestPlayers { eventTeamPlayerId isCaptain isViceCaptain totalPoints }
        }
      }
    }
    """
    try:
        payload = {"operationName": "UserContest", "query": team_query, "variables": {"args": {"id": int(entry['userContestId'])}}}
        res = session.post(base_url, json=payload, timeout=10).json()

        # Handle Expiration
        if "errors" in res and res['errors'][0].get('message') == "Unauthorized":
            with lock:
                new_token = get_automated_token(USERNAME, PASSWORD)
                if new_token:
                    auth_token = new_token
                    session.headers.update({"Authorization": f"Bearer {auth_token}"})
                    res = session.post(base_url, json=payload, timeout=10).json()

        user_team_data = res.get('data', {}).get('user_contest')
        if not user_team_data: return

        user_team = user_team_data['team']
        enriched_players = []
        for p in user_team['players']:
            pid = str(p['eventTeamPlayerId'])
            meta = player_mappings.get(cid, {}).get(pid, {})
            enriched_players.append({
                 "name": meta.get("displayName", f"Unknown ({pid})"),
                        "firstName": meta.get("firstName", f"Unknown ({pid})"),
                        "lastName": meta.get("lastName", f"Unknown ({pid})"),
                        "position": meta.get("position"),
                        "team": meta.get("team"),
                        "cost": meta.get("cost"),
                        "isCaptain": p['isCaptain'],
                        'isViceCaptain': p['isViceCaptain'],
                        "pointsInThisContest": p['totalPoints'],
                        "seasonGoals": meta.get("goals"),
                        "seasonAssists": meta.get("assists")
            })

        final_item = {
            "userContestId": entry['userContestId'],
            "Gameweek": entry['gameweek_name'],
            "Event": entry.get('event_name'),
            "Season": entry.get('event_season'),
            "CompletedAt": entry.get('completedAt'),
            "ContestEntries": entry.get('userContestCount'),
            "EntryFees": entry.get('entryFees'),
            "Rank": entry['rank'],
            "Username": entry['username'],
            "Points": entry['totalPoints'],
            "Formation": user_team['formation'],
            "Players": enriched_players
        }

        with lock:
            processed_data.append(final_item)
            if len(processed_data) % 25 == 0:
                save_progress_atomically(processed_data)
                print(f"  [Progress] Saved {len(processed_data)} entries...")

    except Exception as e:
        print(f"  [Error] {entry['username']}: {e}")

# Run in parallel
print(f"Starting parallel processing of {len(to_process)} items...")
with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
    executor.map(process_entry, to_process)

# --- 4. EXPORT ---
save_progress_atomically(processed_data)

if os.path.exists(PROGRESS_FILE):
    pd.read_json(PROGRESS_FILE).to_csv("fantasy_data_complete.csv", index=False)
    print("\nExport Complete: fantasy_data_complete.csv")