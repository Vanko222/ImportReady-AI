"""Consumer Web UI package (Streamlit presentation layer).

Modules:

* ``i18n`` - fixed EN/ZH translation dictionary (UI labels only)
* ``state`` - session state, session-only BYOK credentials, provider resolution
* ``pipeline`` - adapter to the accepted deterministic engines (no logic)
* ``presenters`` - pure canonical-result -> view-data mapping
* ``components`` - Streamlit rendering helpers
* ``resources`` - cached, credential-free repository/service accessors
* ``app`` - the page itself (entrypoint: repository-root ``app.py``)
"""
