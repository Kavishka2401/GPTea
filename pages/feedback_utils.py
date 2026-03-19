# feedback_utils.py
# Place this file in your project root (same level as your pages/ folder).
# Every page imports and calls render_feedback_widget(PAGE_ID, ICON_PATH).

import json
import base64
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from datetime import datetime

FEEDBACK_DIR = Path("feedback")

# Maps page_id → json filename
FEEDBACK_FILES = {
    "disease_detection": "disease_feedback.json",
    "yield_prediction":  "yield_feedback.json",
    "chatbot":           "chatbot_feedback.json",
}


def _save_feedback(page_id: str, stars: int, comment: str) -> None:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    feedback_file = FEEDBACK_DIR / FEEDBACK_FILES[page_id]

    if feedback_file.exists():
        try:
            with open(feedback_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except (json.JSONDecodeError, OSError):
            data = []
    else:
        data = []

    data.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "page":      page_id,
        "stars":     stars,
        "comment":   comment.strip(),
    })

    with open(feedback_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def render_feedback_widget(page_id: str, icon_path: str) -> None:
    """
    Call this once per page, BEFORE any other st.* widgets.

    page_id   : one of 'disease_detection', 'yield_prediction', 'chatbot'
    icon_path : absolute or relative path to the button icon PNG
    """

    # ── 1. Read & save any incoming feedback from the URL param ──────────────
    _fb_raw = st.query_params.get("fb")

    if _fb_raw:
        try:
            _payload  = json.loads(_fb_raw)
            _page     = str(_payload.get("page", page_id))
            # Only save if the param belongs to this page
            if _page == page_id and _page in FEEDBACK_FILES:
                _save_feedback(
                    page_id = _page,
                    stars   = int(_payload.get("stars",   0)),
                    comment = str(_payload.get("comment", "")),
                )
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        st.query_params.clear()

    # ── 2. Load icon ──────────────────────────────────────────────────────────
    with open(icon_path, "rb") as _f:
        _icon_b64 = base64.b64encode(_f.read()).decode()

    # ── 3. Render the floating widget ─────────────────────────────────────────
    components.html(
        f"""
        <script>
        (function() {{
          const doc  = window.parent.document;
          const PAGE = '{page_id}';

          // Remove widget from a different page; skip re-injection on same page
          const existing = doc.getElementById('gptea-feedback-trigger-wrapper');
          if (existing) {{
            if (existing.getAttribute('data-page') !== PAGE) {{
              existing.remove();
            }} else {{
              return;
            }}
          }}

          /* ── CSS ── */
          const style = doc.createElement('style');
          style.textContent = `
            #gptea-feedback-trigger {{
              position: fixed; bottom: 55px; right: 55px; z-index: 999999;
              display: flex; align-items: center; gap: 8px;
              background: #ffffff; color: #000000;
              border: 2px solid #1f5a3f; border-radius: 999px;
              padding: 10px 20px 10px 12px;
              font-size: 15px; font-weight: 700; cursor: pointer;
              box-shadow: 0 4px 18px rgba(31,90,63,0.38);
              transition: background 0.2s, transform 0.15s;
              letter-spacing: 0.3px; font-family: sans-serif;
            }}
            #gptea-feedback-trigger:hover {{ background: #f0faf4; transform: scale(1.04); }}
            #gptea-feedback-trigger img {{
              width: 28px; height: 28px; border-radius: 50%; object-fit: cover;
            }}
            #gptea-feedback-panel {{
              position: fixed; bottom: 120px; right: 55px; z-index: 999998;
              width: 320px; background: #ffffff; border-radius: 20px;
              box-shadow: 0 8px 40px rgba(31,90,63,0.22), 0 2px 8px rgba(0,0,0,0.10);
              padding: 24px 22px 20px 22px;
              display: none; flex-direction: column; gap: 14px;
              border: 1px solid rgba(31,90,63,0.13); font-family: sans-serif;
              animation: gptea-slidein 0.22s cubic-bezier(.4,0,.2,1);
            }}
            #gptea-feedback-panel.open {{ display: flex; }}
            @keyframes gptea-slidein {{
              from {{ opacity:0; transform: translateY(16px) scale(0.97); }}
              to   {{ opacity:1; transform: translateY(0) scale(1); }}
            }}
            .fb-header {{ display: flex; align-items: center; justify-content: space-between; }}
            .fb-title  {{ font-size: 16px; font-weight: 800; color: #1f5a3f; margin: 0; }}
            .fb-close  {{ background: none; border: none; font-size: 20px; color: #888; cursor: pointer; line-height: 1; }}
            .fb-close:hover {{ color: #1f5a3f; }}
            .fb-sub    {{ font-size: 13px; color: #666; margin: 0; }}
            .fb-stars  {{ display: flex; gap: 6px; justify-content: center; margin: 4px 0; }}
            .fb-stars span {{
              font-size: 32px; cursor: pointer; color: #d0d0d0;
              transition: color 0.15s, transform 0.12s; user-select: none; line-height: 1;
            }}
            .fb-stars span:hover,
            .fb-stars span.active {{ color: #f5a623; transform: scale(1.18); }}
            #gptea-fb-text {{
              width: 100%; min-height: 72px;
              border: 1px solid rgba(31,90,63,0.22); border-radius: 10px;
              padding: 9px 11px; font-size: 13px; color: #222;
              resize: vertical; outline: none; font-family: inherit;
              box-sizing: border-box; transition: border-color 0.15s;
            }}
            #gptea-fb-text:focus {{ border-color: #1f5a3f; }}
            #gptea-fb-text::placeholder {{ color: #aaa; }}
            #gptea-fb-submit {{
              width: 100%; background: #1f5a3f; color: #fff;
              border: none; border-radius: 999px; padding: 10px 0;
              font-size: 14px; font-weight: 700; cursor: pointer;
              transition: background 0.18s, transform 0.12s; font-family: inherit;
            }}
            #gptea-fb-submit:hover {{ background: #184530; transform: scale(1.02); }}
            #gptea-fb-thankyou {{
              display: none; text-align: center;
              color: #1f5a3f; font-weight: 700; font-size: 15px; padding: 8px 0 4px;
            }}
          `;
          doc.head.appendChild(style);

          /* ── HTML ── */
          const wrapper = doc.createElement('div');
          wrapper.id = 'gptea-feedback-trigger-wrapper';
          wrapper.setAttribute('data-page', PAGE);
          wrapper.innerHTML = `
            <button id="gptea-feedback-trigger">
              <img src="data:image/png;base64,{_icon_b64}" alt="GPTea" />
              Feedback
            </button>
            <div id="gptea-feedback-panel">
              <div class="fb-header">
                <p class="fb-title">🍃 Share your feedback</p>
                <button class="fb-close" id="gptea-fb-close">✕</button>
              </div>
              <p class="fb-sub">How would you rate your experience? (optional)</p>
              <div class="fb-stars" id="gptea-fb-stars">
                <span data-val="1">★</span><span data-val="2">★</span>
                <span data-val="3">★</span><span data-val="4">★</span>
                <span data-val="5">★</span>
              </div>
              <textarea id="gptea-fb-text" placeholder="Write your feedback here... (optional)"></textarea>
              <button id="gptea-fb-submit">Submit Feedback</button>
              <div id="gptea-fb-thankyou">✅ Thank you for your feedback!</div>
            </div>
          `;
          doc.body.appendChild(wrapper);

          /* ── Logic ── */
          let rating = 0;
          const trigger   = doc.getElementById('gptea-feedback-trigger');
          const panel     = doc.getElementById('gptea-feedback-panel');
          const closeBtn  = doc.getElementById('gptea-fb-close');
          const stars     = doc.querySelectorAll('#gptea-fb-stars span');
          const submitBtn = doc.getElementById('gptea-fb-submit');
          const thankyou  = doc.getElementById('gptea-fb-thankyou');
          const textarea  = doc.getElementById('gptea-fb-text');
          const sub       = doc.querySelector('.fb-sub');
          const starsDiv  = doc.getElementById('gptea-fb-stars');

          trigger.addEventListener('click', () => panel.classList.toggle('open'));
          closeBtn.addEventListener('click', () => panel.classList.remove('open'));

          stars.forEach(s => {{
            s.addEventListener('click', () => {{
              rating = parseInt(s.dataset.val);
              stars.forEach(x => x.classList.toggle('active', parseInt(x.dataset.val) <= rating));
            }});
          }});

          submitBtn.addEventListener('click', () => {{
            // Include page_id in the payload so Python knows which file to write
            const payload = JSON.stringify({{
              page:    PAGE,
              stars:   rating,
              comment: textarea.value.trim()
            }});

            const url = new URL(window.parent.location.href);
            url.searchParams.set('fb', payload);
            window.parent.history.pushState({{}}, '', url.toString());
            window.parent.dispatchEvent(new PopStateEvent('popstate', {{ state: {{}} }}));

            starsDiv.style.display  = 'none';
            textarea.style.display  = 'none';
            submitBtn.style.display = 'none';
            sub.style.display       = 'none';
            thankyou.style.display  = 'block';

            setTimeout(() => {{
              panel.classList.remove('open');
              setTimeout(() => {{
                rating = 0;
                stars.forEach(x => x.classList.remove('active'));
                textarea.value          = '';
                starsDiv.style.display  = '';
                textarea.style.display  = '';
                submitBtn.style.display = '';
                sub.style.display       = '';
                thankyou.style.display  = 'none';
              }}, 400);
            }}, 2200);
          }});
        }})();
        </script>
        """,
        height=0,
    )