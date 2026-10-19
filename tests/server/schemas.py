"""The five JSON schemas of the structured-output tests (lab-api): CPU grammar tests and the whole-model HTTP test.
Each entry: name, schema, a user prompt for the real model, a valid example (the CPU tests walk it token by token), an
invalid example and `expect` (a check on the real model's parsed answer)."""

PERSON = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
        "email": {"type": "string"},
        "address": {"type": "object",
                    "properties": {"street": {"type": "string"}, "city": {"type": "string"},
                                   "geo": {"type": "object", "properties": {"lat": {"type": "number"},
                                                                            "lon": {"type": "number"}},
                                           "required": ["lat", "lon"], "additionalProperties": False}},
                    "required": ["street", "city", "geo"], "additionalProperties": False},
        "tags": {"type": "array", "items": {"type": "string"}},
        "active": {"type": "boolean"}},
    "required": ["name", "age", "email", "address", "tags", "active"], "additionalProperties": False}

TICKET = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "priority": {"enum": ["low", "medium", "high", "urgent"]},
        "status": {"type": "string", "enum": ["open", "in_progress", "closed"]},
        "assignee": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "items": {"type": "array", "minItems": 1,
                  "items": {"type": "object", "properties": {"task": {"type": "string"},
                                                              "hours": {"type": "number"}},
                            "required": ["task", "hours"], "additionalProperties": False}}},
    "required": ["title", "priority", "status", "assignee", "items"], "additionalProperties": False}

RECIPE = {
    "type": "object",
    "properties": {
        "dish": {"type": "string"},
        "servings": {"type": "integer"},
        "ingredients": {"type": "array", "minItems": 2, "maxItems": 12,
                        "items": {"type": "object",
                                  "properties": {"item": {"type": "string"}, "amount": {"type": "number"},
                                                 "unit": {"enum": ["g", "ml", "tsp", "tbsp", "piece"]}},
                                  "required": ["item", "amount", "unit"], "additionalProperties": False}},
        "steps": {"type": "array", "items": {"type": "string"}, "minItems": 2},
        "vegetarian": {"type": "boolean"}},
    "required": ["dish", "servings", "ingredients", "steps", "vegetarian"], "additionalProperties": False}

VERDICT = {
    "type": "object",
    "properties": {
        "verdict": {"enum": ["yes", "no", "unsure"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasons": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "note": {"type": ["string", "null"]}},
    "required": ["verdict", "confidence", "reasons", "note"], "additionalProperties": False}

TRIP = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "legs": {"type": "array", "minItems": 2,
                 "items": {"type": "object",
                           "properties": {"from": {"type": "string"}, "to": {"type": "string"},
                                          "mode": {"enum": ["train", "plane", "bus", "walk"]},
                                          "waypoints": {"type": "array",
                                                        "items": {"type": "array", "items": {"type": "number"},
                                                                  "minItems": 2, "maxItems": 2}}},
                           "required": ["from", "to", "mode", "waypoints"], "additionalProperties": False}},
        "total_km": {"type": "number"}},
    "required": ["title", "legs", "total_km"], "additionalProperties": False}

CASES = [
    {"name": "person", "schema": PERSON,
     "prompt": "Invent a person who lives in Lyon, France, and describe them as JSON.",
     "valid": {"name": "Ana Silva", "age": 34, "email": "ana@example.org",
               "address": {"street": "12 Rue Merciere", "city": "Lyon", "geo": {"lat": 45.76, "lon": 4.83}},
               "tags": ["chef", "runner"], "active": True},
     "invalid": {"name": "Ana", "age": "34"},
     "expect": lambda o: True},
    {"name": "ticket", "schema": TICKET,
     "prompt": "Create a support ticket for a printer that jams on every page; it is blocking the whole office, "
               "nobody is assigned yet, and it needs two tasks.",
     "valid": {"title": "Printer jams", "priority": "high", "status": "open", "assignee": None,
               "items": [{"task": "Check rollers", "hours": 1.5}, {"task": "Replace tray", "hours": 2}]},
     "invalid": {"title": "x", "priority": "critical", "status": "open", "assignee": None, "items": []},
     "expect": lambda o: o["assignee"] is None and o["priority"] in ("high", "urgent") and len(o["items"]) >= 2},
    {"name": "recipe", "schema": RECIPE,
     "prompt": "Give me a simple vegetarian pancake recipe for 4 people.",
     "valid": {"dish": "Pancakes", "servings": 4,
               "ingredients": [{"item": "flour", "amount": 250, "unit": "g"},
                               {"item": "milk", "amount": 500, "unit": "ml"},
                               {"item": "egg", "amount": 2, "unit": "piece"}],
               "steps": ["Whisk everything.", "Fry in a pan."], "vegetarian": True},
     "invalid": {"dish": "Pancakes", "servings": 4, "ingredients": [{"item": "flour", "amount": 250, "unit": "kg"}],
                 "steps": ["a"], "vegetarian": True},
     "expect": lambda o: o["servings"] == 4 and o["vegetarian"] is True},
    {"name": "verdict", "schema": VERDICT,
     "prompt": "Is the Earth larger than the Moon? Give a verdict, your confidence and up to three reasons.",
     "valid": {"verdict": "yes", "confidence": 0.99, "reasons": ["Earth's diameter is about 12,700 km",
                                                                  "The Moon's is about 3,500 km"], "note": None},
     "invalid": {"verdict": "maybe", "confidence": 2, "reasons": [], "note": 3},
     "expect": lambda o: o["verdict"] == "yes" and o["confidence"] >= 0.8},
    {"name": "trip", "schema": TRIP,
     "prompt": "Plan a trip Paris -> Lyon by train and Lyon -> Nice by plane, with approximate coordinates "
               "[lat, lon] of one waypoint per leg.",
     "valid": {"title": "South of France", "legs": [
         {"from": "Paris", "to": "Lyon", "mode": "train", "waypoints": [[47.0, 4.8]]},
         {"from": "Lyon", "to": "Nice", "mode": "plane", "waypoints": [[44.0, 6.0], [43.9, 7.0]]}], "total_km": 930.5},
     "invalid": {"title": "x", "legs": [{"from": "a", "to": "b", "mode": "boat", "waypoints": [[1]]}], "total_km": 1},
     "expect": lambda o: o["legs"][0]["mode"] == "train" and o["legs"][1]["mode"] == "plane"},
]

WEATHER_TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "city name"},
            "days": {"type": "integer", "description": "forecast days"},
            "units": {"enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "get_time", "description": "Local time in a timezone.",
        "parameters": {"type": "object", "properties": {"tz": {"type": "string", "description": "IANA zone"}},
                       "required": ["tz"]}}},
]
