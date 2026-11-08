"""
reviewer_dashboard.py

Run:

streamlit run reviewer_dashboard.py
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(
    page_title="Roman Candidate Review",
    layout="wide"
)

st.title(
    "Roman Exoplanet Candidate Review"
)

candidate_dir = (
    Path("output/candidates")
)

candidate_dir.mkdir(
    parents=True,
    exist_ok=True
)

candidate_files = list(
    candidate_dir.glob("*.json")
)

if len(candidate_files) == 0:

    st.warning(
        "No candidates found."
    )

    st.stop()

candidate_ids = [
    x.stem
    for x in candidate_files
]

selected_candidate = st.selectbox(

    "Candidate",

    candidate_ids
)

filepath = (
    candidate_dir
    /
    f"{selected_candidate}.json"
)

with open(filepath) as f:

    data = json.load(f)

left, right = st.columns(2)

with left:

    st.subheader(
        "Candidate Summary"
    )

    st.json(data)

with right:

    st.subheader(
        "Reviewer Actions"
    )

    reviewer = st.text_input(
        "Reviewer"
    )

    decision = st.radio(

        "Decision",

        [
            "Accepted",
            "Rejected",
            "Needs Further Study",
            "Possible Binary Lens"
        ]
    )

    comments = st.text_area(
        "Comments"
    )

    if st.button(
        "Submit Review"
    ):

        review = {

            "reviewer":
                reviewer,

            "status":
                decision,

            "comments":
                comments
        }

        data.setdefault(
            "reviews",
            []
        )

        data["reviews"].append(
            review
        )

        data["status"] = decision

        with open(
            filepath,
            "w"
        ) as f:

            json.dump(
                data,
                f,
                indent=4
            )

        st.success(
            "Review Saved"
        )

st.subheader(
    "Review History"
)

for review in data.get(
    "reviews",
    []
):

    st.markdown(
        f"""
        **Reviewer:** {review['reviewer']}

        **Status:** {review['status']}

        **Comments:** {review['comments']}

        ---
        """
    )