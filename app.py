import streamlit as st

# 1) Define Page objects:
page_home = st.Page("other_pages/home.py", title="Home", icon="👨‍🌾")
page_demo = st.Page("other_pages/demo.py", title="Demo", icon="🤖")
page_metrics = st.Page("other_pages/metrics.py", title="Metrics", icon="📈")
page_glossary = st.Page("other_pages/glossary.py", title="Glossary", icon="📖")
page_refs = st.Page("other_pages/references.py", title="References", icon="📚")

# 2) Configure the navigation menu:
pages = [page_home, page_demo, page_metrics, page_glossary, page_refs]
selected_page = st.navigation(pages)

# 4) Run the selected page:
selected_page.run()