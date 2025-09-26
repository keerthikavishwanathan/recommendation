# app.py

import streamlit as st
import json
import torch
import faiss
import textwrap
from search_engine2 import (
    UniversityEmbedder,
    load_mapping,
    get_university_name,
    search_user_query,
)

# -------------------- Load Models and Data --------------------
@st.cache_resource
def load_resources():
    """Loads all the necessary models and data files."""
    index = faiss.read_index("CDS_vectors.index")
    mapping = load_mapping("CDS_vectors_mapping.json")
    model = UniversityEmbedder()
    # Ensure model loads on CPU if no GPU is available
    model.load_state_dict(torch.load("university_embedder2.pth", map_location="cpu"))
    model.eval()
    
    with open("CDS1.json", "r", encoding="utf-8") as f:
        full_university_data = json.load(f)
        
    university_details_map = {}
    for uni_data in full_university_data:
        name = get_university_name(uni_data)
        if name:
            university_details_map[name.lower()] = uni_data
            
    return model, index, mapping, university_details_map

# -------------------- Streamlit Display Function --------------------
def display_university_details_st(details):
    """
    Formats and displays the details of a selected university in Streamlit.
    Replaces the original print-based function.
    """
    # --- 1. Acceptance Rate Calculation & Display ---
   

    # --- 2. Admission Factors Display ---
    admissions_factors = details.get("admissions_factors")
    if isinstance(admissions_factors, dict):
        very_important_factors = [
            key.replace('_', ' ').title() 
            for key, value in admissions_factors.items() 
            if value == "Very Important"
        ]
        if very_important_factors:
            st.subheader("⭐ Essential Factors for Admission")
            st.info(", ".join(very_important_factors))

    # --- 3. Financial Aid URL Display ---
    financial_aid_url = details.get("financial_aid")
    if financial_aid_url and isinstance(financial_aid_url, str) and financial_aid_url.startswith('http'):
        st.subheader("💰 Financial Aid")
        st.markdown(f"[Click here for fee structure and financial aid]({financial_aid_url})")

    # --- 4. Display all other details ---
    st.subheader("📋 Other Details")
    handled_keys = {
        "financial_aid", "admissions_factors",
        "first_time_men_applied", "first_time_women_applied",
        "first_time_men_admitted", "first_time_women_admitted",
        "name_of_university", "Name of University/College", "Name of College/University", "name", "college_name"
    }
    
    other_details = {}
    for key, value in details.items():
        if (key in handled_keys or value is None or value == "" or 
            (isinstance(value, list) and not value) or "sat" in key.lower() or 
            "act" in key.lower() or "gpa" in key.lower()):
            continue
        
        formatted_key = key.replace('_', ' ').replace('gpa', 'GPA').title()
        
        if isinstance(value, list):
            value_str = ", ".join(map(str, value))
            other_details[formatted_key] = value_str
        elif not isinstance(value, dict):
             other_details[formatted_key] = value

    st.json(other_details)


# -------------------- Streamlit UI --------------------
def main():
    st.set_page_config(page_title="University Recommendation System", layout="wide")
    st.title("🎓 University Recommendation System")
    st.write("Find universities that match your academic profile using a cosine similarity based recommendation engine.")

    # Load cached resources
    model, index, mapping, university_details_map = load_resources()

    # Initialize session state to store search results
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Sidebar inputs
    with st.sidebar:
        st.header("Enter Your Details")
        user_gpa = st.number_input("Your GPA (0.0 - 4.0)", min_value=0.0, max_value=4.0, step=0.1, value=3.5)

        sat_or_act = st.radio("Select your test type", ["SAT", "ACT"])
        user_sat, user_act = None, None
        if sat_or_act == "SAT":
            user_sat = st.number_input("Your SAT score (400-1600)", min_value=400, max_value=1600, step=10, value=1200)
        else:
            user_act = st.number_input("Your ACT score (1-36)", min_value=1, max_value=36, step=1, value=25)

        user_text = st.text_input("Preferences (e.g., location, major)", "")

        if st.button("🔍 Get Recommendations", type="primary"):
            with st.spinner("Finding matching universities..."):
                ambitious, practical, safe = search_user_query(
                    model=model,
                    index=index,
                    mapping=mapping,
                    university_details_map=university_details_map, # Pass the map here
                    gpa=user_gpa,
                    sat=user_sat,
                    act=user_act,
                    text=user_text,
                    top_k=30,
                    debug=False,
                )
                # Store results in session state
                st.session_state.results = (ambitious, practical, safe)

    # Display results if they exist in the session state
    if st.session_state.results:
        ambitious_unis, practical_unis, safe_unis = st.session_state.results
        
        st.header("🔍 Your Recommended Universities")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🌠 Ambitious")
            if ambitious_unis:
                for name, sim, details in ambitious_unis:
                    with st.expander(f"{name} (Similarity: {sim:.3f})"):
                        display_university_details_st(details)
            else:
                st.write("No results in this category.")
        
        with col2:
            st.subheader("🎯 Practical")
            if practical_unis:
                for name, sim, details in practical_unis:
                    with st.expander(f"{name} (Similarity: {sim:.3f})"):
                        display_university_details_st(details)
            else:
                st.write("No results in this category.")

        with col3:
            st.subheader("✅ Safe")
            if safe_unis:
                for name, sim, details in safe_unis:
                    with st.expander(f"{name} (Similarity: {sim:.3f})"):
                        display_university_details_st(details)
            else:
                st.write("No results in this category.")
    else:
        st.info("Enter your details in the sidebar and click 'Get Recommendations' to begin.")

if __name__ == "__main__":
    main()
