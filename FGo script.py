import requests
import json
import time
import pandas as pd
import os

# 071 401 2550
# poniZION1
# --- 1. CONFIGURATION & AUTOMATED LOGIN ---
# Enter your credentials here to avoid manual token updates
raw_username = input("Please paste your username: ")
raw_password = input("Please paste your password: ")

USERNAME = raw_username.strip()
PASSWORD = raw_password.strip()


base_url = "https://api.fantasygo.io/public"
cdn_url = "https://cloudcdn.fantasygo.io/contest-event-team-players/contest_event_team_players_{}.json"

def get_automated_token(username, password):
    """Logs into FantasyGo using username and retrieves a fresh Bearer token."""
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
    # Note: Ensure the username format matches what the site expects (e.g., spaces or no spaces)
    login_vars = {"args": {"mobile": username, "password": password}}

    try:
        response = requests.post(base_url, json={
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
        print("Using Auth Token:", token)
        return token


    except Exception as e:
        print(f"Error during login: {e}")
        return None

# Authenticate automatically
auth_token = get_automated_token(USERNAME, PASSWORD)

if not auth_token:
    print("Could not proceed without a valid token. Check credentials.")
    # Fallback to manual if automated fails
    raw_token = input("Fallback: Please paste your Bearer Token manually: ")
    auth_token = raw_token.replace("Bearer ", "").strip()

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {auth_token}",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
    "Origin": "https://app.fantasygo.io",
    "Referer": "https://app.fantasygo.io/"
}


# File paths
LEADERBOARD_CACHE = "raw_leaderboards.json"
PROGRESS_FILE = "processed_results.json"

# --- 2. STAGE 1: DISCOVERY (FETCH GAMEWEEKS & LEADERBOARD METADATA) ---
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
    # Fetch all possible Gameweeks (limit 3000)
    vars_contests = {"args": {"eventTypes": ["englishPremierLeague"], "gameTypes": ["fantasy"], "limit": 3000, "offset": 0}}

    try:
        res_raw = requests.post(base_url, headers=headers, json={"operationName": "EndedContests", "query": ended_contests_query, "variables": vars_contests})
        res = res_raw.json()

        if "errors" in res:
            print(f"API Error in Stage 1: {res['errors'][0].get('message')}")
            gameweeks = []
        else:
            all_contests = res['data']['ended_contests']['contests']
            gameweeks = [c for c in all_contests if not c['isSingleMatchContest']]
            print(f"Found {len(gameweeks)} Gameweek contests.")
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
        # Fetch up to 3000 entries per leaderboard
        lb_vars = {"args": {"contestId": int(gw['id']), "limit": 3000, "offset": 0, "entryFee": 10}}
        lb_res = requests.post(base_url, headers=headers, json={"operationName": "LeaderboardForContest", "query": lb_query, "variables": lb_vars}).json()

        entries = lb_res.get('data', {}).get('leaderboard', {}).get('items', [])

        # Clean the entryFees list (e.g. [10] -> "10")
        raw_fees = gw.get('entryFees', [])
        clean_fees = ", ".join(map(str, raw_fees)) if isinstance(raw_fees, list) else str(raw_fees)

        for entry in entries:
            entry['gameweek_name'] = gw['name']
            entry['gameweek_id'] = gw['id']
            entry['entryFees'] = clean_fees
            entry['userContestCount'] = gw.get('userContestCount')
            entry['completedAt'] = gw.get('completedAt')
            entry['event_name'] = gw.get('event', {}).get('name')
            entry['event_season'] = gw.get('event', {}).get('season')
            leaderboard_data.append(entry)
        time.sleep(0.3)

    if leaderboard_data:
        with open(LEADERBOARD_CACHE, 'w') as f:
            json.dump(leaderboard_data, f, indent=4)
        print(f"Done! Cached {len(leaderboard_data)} entries.")

# --- 3. STAGE 2: RESUMABLE EXTRACTION ---
print("\n--- STAGE 2: Extraction Phase ---")

if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, 'r') as f:
        processed_data = json.load(f)
else:
    processed_data = []

processed_ids = {item['userContestId'] for item in processed_data}
print(f"Resuming: {len(processed_ids)} entries already processed.")

if not os.path.exists(LEADERBOARD_CACHE):
    print("Error: No leaderboard cache found. Please complete Stage 1.")
else:
    with open(LEADERBOARD_CACHE, 'r') as f:
        master_list = json.load(f)

    to_process = [item for item in master_list if item['userContestId'] not in processed_ids]

    # Cache for player mappings per contest
    player_mappings = {}
    consecutive_errors = 0
    failed_this_session = [] # Track items that failed during this run to retry later

    def get_stat(stats_list, stat_type):
        """Helper to extract a specific value from the statistics array."""
        if not stats_list: return "0"
        for s in stats_list:
            if s.get('type') == stat_type:
                return s.get('value')
        return "0"

    if not to_process:
        print("Everything is already processed!")
    else:

        for entry in to_process:
            cid = str(entry['gameweek_id'])

            # Fetch Player Mapping if not already cached
            if cid not in player_mappings:
                print(f"  [New Contest] Fetching detailed player metadata for ID {cid}...")
                try:
                    map_res = requests.get(cdn_url.format(cid))
                    if map_res.status_code == 200:
                        raw_players = map_res.json()
                        player_mappings[cid] = {}
                        # Map based on the flat structure provided in query
                        for p in raw_players:
                            pid = str(p['id'])
                            player_mappings[cid][pid] = {
                                "displayName": p.get('displayName'),
                                "firstName": p.get('firstName'),
                                "lastName": p.get('lastName'),
                                "position": p.get('position'),
                                "cost": p.get('cost'),
                                "team": p.get('team', {}).get('name'),
                                "goals": get_stat(p.get('statistics', []), "goalsScored"),
                                "assists": get_stat(p.get('statistics', []), "assists")
                            }
                    else:
                        player_mappings[cid] = {}
                except Exception as e:
                    print(f"    Warning: Could not fetch player names for contest {cid}: {e}")
                    player_mappings[cid] = {}

            print(f"  [{len(processed_data)}/{len(master_list)}] Fetching: {entry['username']} ({entry['gameweek_name']})")

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
                team_res_raw = requests.post(base_url, headers=headers, json={
                    "operationName": "UserContest", "query": team_query, "variables": {"args": {"id": int(entry['userContestId'])}}
                })
                team_res = team_res_raw.json()


                # Auto Re-Authentication if token expires mid-loop
                if "errors" in team_res and team_res['errors'][0].get('message') == "Unauthorized":
                    print("\n[!] TOKEN EXPIRED. Re-authenticating automatically...")
                    new_token = get_automated_token(USERNAME, PASSWORD)
                    if new_token:
                        headers["Authorization"] = f"Bearer {new_token}"
                        # Retry the current selection
                        team_res = requests.post(base_url, headers=headers, json={
                            "operationName": "UserContest", "query": team_query, "variables": {"args": {"id": int(entry['userContestId'])}}
                        }).json()
                    else:
                        print("Re-authentication failed. Please check network/credentials.")
                        break


                user_team_data = team_res.get('data', {}).get('user_contest')
                if not user_team_data:
                    continue

                user_team = user_team_data['team']

                # Enrich players with metadata from our mapping
                enriched_players = []
                for p in user_team['players']:
                    pid = str(p['eventTeamPlayerId'])
                    meta = player_mappings[cid].get(pid, {})
                    # print(player_mappings)
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

                processed_data.append({
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
                })

                # Incremental Save
                with open(PROGRESS_FILE, 'w') as f:
                    json.dump(processed_data, f, indent=4)

                time.sleep(0.2)

            except Exception as e:
                print(f"    Error on {entry['username']}: {e}")
                continue

# --- 4. EXPORT TO CSV ---
if os.path.exists(PROGRESS_FILE):
    df = pd.read_json(PROGRESS_FILE)
    df.to_csv("fantasy_data_complete.csv", index=False)
    print("\nFinal data synchronized to fantasy_data_complete.csv")
    print(df.head())