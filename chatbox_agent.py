import streamlit as st
import streamlit.components.v1 as components

# Mã iframe đơn giản (không bao gồm <script>)
iframe_code = """
<iframe id="JotFormIFrame-0196a4f69c447fa1b3f692b92971658a35c2" title="Sales AI Agent"
  allowtransparency="true"
  allow="geolocation; microphone; camera; fullscreen"
  src="https://agent.jotform.com/0196a4f69c447fa1b3f692b92971658a35c2?embedMode=iframe&background=1&shadow=1"
  frameborder="0" style="
    min-width:100%;
    max-width:100%;
    height:688px;
    border:none;
    width:100%;
  " scrolling="no">
</iframe>
"""

# Hiển thị iframe trong Streamlit
components.html(iframe_code, height=700)
