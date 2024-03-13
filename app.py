import streamlit as st
from st_pages import Page, show_pages, add_page_title

show_pages(
    [
        Page(
            "other_pages/home.py",
            "Home",
            "👨‍🌾",
        ),
        Page("other_pages/demo.py", "Demo", "🤖"),
        Page("other_pages/metrics.py", "Metrics", "📈"),
        Page("other_pages/glossary.py", "Glossary", "📖"),
    ]
)
