# search_engine2.py

import json
import numpy as np
import torch
import torch.nn as nn
import faiss
from transformers import BertTokenizer, BertModel
from typing import List
import textwrap


# -------------------- Embedder Class --------------------
class UniversityEmbedder(nn.Module):
    """
    Numeric-only embedder (matches your checkpoint):
      - Input: 15 numeric features (SAT, ACT, GPA bins, avg GPA)
      - Output: projected embedding (proj_dim)
    """

    def __init__(self, num_input_dim: int = 15, proj_dim: int = 128, device: str = None):
        super().__init__()
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Numeric MLP
        self.num_mlp = nn.Sequential(
            nn.Linear(num_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, proj_dim)
        )

        self.to(self.device)

    def normalize(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Normalization with spread:
        SAT / 800, ACT / 18, GPA bins / 25, avg GPA / 4
        """
        sat = feats[:, 0:3] / 800.0
        act = feats[:, 3:6] / 18.0
        gpa_bins = feats[:, 6:14] / 25.0  # percentage values exaggerated
        avg_gpa = feats[:, 14:15] / 4.0    # GPA scale 0–4
        return torch.cat([sat, act, gpa_bins, avg_gpa], dim=1)

    def forward(self, num_features: torch.Tensor) -> torch.Tensor:
        num_features = num_features.to(self.device)
        num_norm = self.normalize(num_features)
        num_proj = self.num_mlp(num_norm)
        return num_proj
    
# -------------------- Load Mapping --------------------
def load_mapping(mapping_path: str):
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------- GPA → bins --------------------
def gpa_to_bins(gpa: float):
    bins = [0.0] * 8
    if gpa >= 3.75: bins[0] = 100.0
    elif 3.50 <= gpa < 3.75: bins[1] = 100.0
    elif 3.25 <= gpa < 3.50: bins[2] = 100.0
    elif 3.00 <= gpa < 3.25: bins[3] = 100.0
    elif 2.50 <= gpa < 3.00: bins[4] = 100.0
    elif 2.00 <= gpa < 2.50: bins[5] = 100.0
    elif 1.00 <= gpa < 2.00: bins[6] = 100.0
    else: bins[7] = 100.0
    return bins


# Add this helper function to both python scripts

def convert_sat_to_act(sat_score: float) -> float:
    """Converts an SAT score to an equivalent ACT score using a simplified concordance table."""
    if sat_score == 0: return 0.0
    # This is a simplified table. For a production system, use a more granular one.
    concordance = {
        1600: 36, 1560: 35, 1520: 34, 1490: 33, 1450: 32, 1420: 31, 1390: 30,
        1360: 29, 1330: 28, 1300: 27, 1260: 26, 1230: 25, 1200: 24, 1160: 23,
        1130: 22, 1100: 21, 1060: 20, 1030: 19, 990: 18, 960: 17, 920: 16, 900: 15,
        880: 14, 850: 13, 820: 12, 800: 11, 780: 10, 750: 9, 720: 8, 700: 7, 650: 6, 600: 5, 550: 4, 500: 3, 400: 2, 300: 1, 200: 0
    }
    # Find the closest SAT score in the table and return its corresponding ACT score
    closest_sat = min(concordance.keys(), key=lambda k: abs(k - sat_score))
    return float(concordance[closest_sat])

# You can create a reverse function as well if needed
def convert_act_to_sat(act_score: float) -> float:
    """Converts an ACT score to an equivalent SAT score."""
    if act_score == 0: return 0.0
    # Simplified reverse table
    concordance = {
        36: 1600, 35: 1560, 34: 1520, 33: 1490, 32: 1450, 31: 1420, 30: 1390,
        29: 1360, 28: 1330, 27: 1300, 26: 1260, 25: 1230, 24: 1200, 23: 1160,
        22: 1130, 21: 1100, 20: 1060, 19: 1030, 18: 990, 17: 960, 16: 920, 15: 900,
        14: 880, 13: 850, 12: 820, 11: 800, 10: 780, 9: 750, 8: 720, 7: 700, 6: 650, 5: 600, 4: 550, 3: 500, 2: 400, 1: 300, 0: 200
    }
    closest_act = min(concordance.keys(), key=lambda k: abs(k - act_score))
    return float(concordance[closest_act])

# -------------------- Vectorize User Query (with debug) --------------------
def vectorize_user_query(model, gpa: float, sat: float = None, act: float = None, text: str = "", debug=False):
    # Convert potential None values to 0.0 immediately.
    user_sat = sat if sat is not None else 0.0
    user_act = act if act is not None else 0.0

    # Apply concordance logic to fill the missing test score
    if user_sat > 0.0 and user_act == 0.0:
        user_act = convert_sat_to_act(user_sat)
    elif user_act > 0.0 and user_sat == 0.0:
        user_sat = convert_act_to_sat(user_act)
        
    # Create a synthetic score range for more robust comparison
    sat_spread = 30
    act_spread = 1
    
    sat_vec = [user_sat - sat_spread, user_sat, user_sat + sat_spread] if user_sat > 0 else [0.0, 0.0, 0.0]
    act_vec = [user_act - act_spread, user_act, user_act + act_spread] if user_act > 0 else [0.0, 0.0, 0.0]

    # These features are not used in the current numeric-only model, but kept for structure
    fee = [0.0]
    ratio = [0.0]
    gpa_bins = gpa_to_bins(gpa)
    avg_gpa = [gpa]

    features = sat_vec + act_vec + gpa_bins + avg_gpa
    features = torch.tensor([features], dtype=torch.float)

    with torch.no_grad():
        user_vec = model(features).cpu().numpy()

    faiss.normalize_L2(user_vec)

    if debug:
        print("\n🟢 DEBUG: Raw numeric features:", features.tolist())
        print("🟢 DEBUG: Normalized numeric features:", model.normalize(features).tolist())
        print("🟢 DEBUG: Final fused + normalized user vector (first 10 dims):", user_vec[0][:10])

    return user_vec

# -------------------- Search (MODIFIED for Streamlit) --------------------
def search_user_query(model, index, mapping, university_details_map, gpa, sat=None, act=None, text="", top_k=30, debug=False):
    query_vec = vectorize_user_query(model, gpa, sat, act, text, debug=debug)
    sims, I = index.search(query_vec, top_k)

    all_results = []
    for j, i in enumerate(I[0]):
        if i < len(mapping):
            uni_name = mapping[i] 
            sim = float(sims[0][j])
            
            # --- THIS IS THE KEY CHANGE ---
            # Fetch the full details for the university
            details = university_details_map.get(uni_name.lower())
            
            if details:
                # Append the full details along with name and similarity
                all_results.append((uni_name, sim, details))

            if debug and j < 5:
                print(f"🟢 DEBUG (Top 5): {uni_name} -> Cosine similarity {sim:.4f}")
        else:
            print(f"⚠️ Warning: Index {i} out of bounds for mapping list.")
            continue

    # Categorize results into three lists based on proximity
    ambitious = all_results[0:10]
    practical = all_results[10:20]
    safe = all_results[20:30]

    return ambitious, practical, safe

# -------------------- Helper functions to display details --------------------
def get_university_name(uni_data):
    """Robustly gets the university name from a dictionary entry."""
    keys_to_check = [
        "name_of_university",
        "Name of University/College",
        "Name of College/University",
        "name",
        "college_name"
    ]
    for key in keys_to_check:
        if key in uni_data:
            return uni_data[key]
    return None

# THIS FUNCTION IS FOR THE TERMINAL. A new version will be made in app.py for Streamlit
def display_university_details(details):
    """Formats and prints the details of a selected university for the terminal."""
    print("\n" + "="*80)
    name = get_university_name(details)
    if name:
        print(f"🎓 Details for {name}")
        print("-" * 80)
    
    # --- 1. Acceptance Rate Calculation & Display ---
    try:
        men_applied = float(details.get("first_time_men_applied") or 0)
        women_applied = float(details.get("first_time_women_applied") or 0)
        men_admitted = float(details.get("first_time_men_admitted") or 0)
        women_admitted = float(details.get("first_time_women_admitted") or 0)
        total_applied = men_applied + women_applied
        total_admitted = men_admitted + women_admitted
        if total_applied > 0:
            acceptance_rate = (total_admitted / total_applied) * 100
            print(f"Acceptance Rate: {acceptance_rate:.2f}% (Calculated from admissions data)")
        else:
            print("Acceptance Rate: Not available (no application data provided)")
        print("-" * 30)
    except (ValueError, TypeError):
        print("Acceptance Rate: Not available (invalid data format)")

    # --- 2. Admission Factors Display ---
    admissions_factors = details.get("admissions_factors")
    if isinstance(admissions_factors, dict):
        very_important_factors = [
            key.replace('_', ' ').title() 
            for key, value in admissions_factors.items() 
            if value == "Very Important"
        ]
        if very_important_factors:
            print("\n--- Essential Factors for Admission ---")
            print("These are the factors considered 'Very Important' by the university:")
            print(textwrap.fill(", ".join(very_important_factors), width=80, initial_indent="  ", subsequent_indent="  "))

    # --- 3. Financial Aid URL Display ---
    financial_aid_url = details.get("financial_aid")
    if financial_aid_url and isinstance(financial_aid_url, str) and financial_aid_url.startswith('http'):
        print("\n--- Financial Aid ---")
        print(f"To know about the fee structure, click this link: {financial_aid_url}")

    # --- 4. Display all other details ---
    print("\n--- Other Details ---")
    handled_keys = {
        "financial_aid", "admissions_factors",
        "first_time_men_applied", "first_time_women_applied",
        "first_time_men_admitted", "first_time_women_admitted",
        "name_of_university", "Name of University/College", "Name of College/University", "name"
    }
    for key, value in details.items():
        if (key in handled_keys or value is None or value == "" or 
            (isinstance(value, list) and not value) or "sat" in key.lower() or 
            "act" in key.lower() or "gpa" in key.lower()):
            continue
        
        formatted_key = key.replace('_', ' ').replace('gpa', 'GPA').title()
        
        if isinstance(value, dict):
            print(f"\n- {formatted_key}:")
            for sub_key, sub_value in value.items():
                formatted_sub_key = sub_key.replace('_', ' ').title()
                print(f"  {formatted_sub_key}: {sub_value}")
        elif isinstance(value, list):
            print(f"\n- {formatted_key}:")
            print(textwrap.fill(", ".join(map(str, value)), width=80, initial_indent="  ", subsequent_indent="  "))
        else:
            print(f"{formatted_key}: {value}")
            
    print("="*80 + "\n")
    
# -------------------- Main (For Terminal use) --------------------
if __name__ == "__main__":
    index = faiss.read_index("CDS_vectors.index")
    mapping = load_mapping("CDS_vectors_mapping.json")
    model = UniversityEmbedder()
    model.load_state_dict(torch.load("university_embedder2.pth"))
    model.eval()

    with open('CDS1.json', 'r', encoding='utf-8') as f:
        full_university_data = json.load(f)

    university_details_map = {}
    for uni_data in full_university_data:
        name = get_university_name(uni_data)
        if name:
            university_details_map[name.lower()] = uni_data

    print("🎓 University Recommendation System (Cosine Similarity)")
    print("-----------------------------------------------------")

    user_gpa = float(input("Enter your GPA (e.g., 3.6): "))
    sat_or_act = input("Do you want to enter SAT or ACT? (type 'sat' or 'act'): ").strip().lower()
    user_sat, user_act = None, None
    if sat_or_act == "sat":
        user_sat = float(input("Enter your SAT score (out of 1600): "))
    elif sat_or_act == "act":
        user_act = float(input("Enter your ACT score (out of 36): "))
    user_text = input("Enter any preference text (location, major, etc.): ").strip()

    # Pass the university_details_map to the search function
    ambitious_unis, practical_unis, safe_unis = search_user_query(
        model, index, mapping, university_details_map,
        gpa=user_gpa, sat=user_sat, act=user_act, text=user_text,
        top_k=30, debug=False
    )

    print("\n\n\n🔍 Recommended Universities")
    print("==============================")

    print("\n🌠 Ambitious Universities (Top 10)")
    print("-----------------------------------")
    # Note: We now unpack three values, but only need two for printing the list
    if ambitious_unis:
        for name, sim, _ in ambitious_unis:
            print(f"{name}  (cosine similarity: {sim:.4f})")
    else:
        print("Not enough results found for this category.")

    print("\n🎯 Practical Universities (Ranks 11-20)")
    print("--------------------------------------")
    if practical_unis:
        for name, sim, _ in practical_unis:
            print(f"{name}  (cosine similarity: {sim:.4f})")
    else:
        print("Not enough results found for this category.")

    print("\n✅ Safe Universities (Ranks 21-30)")
    print("-----------------------------------")
    if safe_unis:
        for name, sim, _ in safe_unis:
            print(f"{name}  (cosine similarity: {sim:.4f})")
    else:
        print("Not enough results found for this category.")

    while True:
        choice = input("\n👉 Enter a university name for full details, or type 'exit' to quit: ").strip()
        if choice.lower() == 'exit':
            print("👋 Goodbye!")
            break
        
        details = university_details_map.get(choice.lower())
        if details:
            display_university_details(details)
        else:
            print(f"❌ University '{choice}' not found. Please ensure the name is spelled correctly.")