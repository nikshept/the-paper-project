import streamlit as st

pg = st.navigation([
    st.Page("homePage.py", title="Home", url_path="home", default=True),
    st.Page("generatePage.py", title="Generate", url_path="generate"),
    st.Page("templatePage.py", title="Template", url_path="template"),
    st.Page("reviewPage.py", title="Review", url_path="review"),
])
pg.run()