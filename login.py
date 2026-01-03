# FantasyGo Automation Scraper (Resumable Jupyter Version)
# Purpose: Fetch EPL Gameweeks -> Get Leaderboards -> Fetch Team Selections
# FEATURE: Automated Login & Progress Saving

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

print("Using Auth Token:", auth_token)